//! The training routes' tests: the queue slot, the forms, the page and its JSON, the actions.
//! `record` builds a run record for other modules' tests too.

use super::fields::*;
use super::routes::*;
use super::view::*;
use super::*;
use crate::backend::detect::{HardwareInfo, apple_gpu};
use crate::config::Config;
use crate::model::ModelFamilyKind;
use crate::resources::export::train::{self as env, TrainPaths};
use crate::server::train_area::{MODELS, TRAIN};
use crate::training::capability::Capability;
use crate::training::compare;
use crate::training::dataset::{Balance, BalanceAxis, Excluded, SceneFilter};
use crate::training::job::Hooks;
use crate::training::job::{BenchSummary, TrainOutput, TrainRequest};
use crate::training::recipe::Augment;
use crate::training::recipe::{self, TrainRecipe};
use crate::training::service::{RunState, TrainingService};
use axum::http::StatusCode;
use std::collections::BTreeMap;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicBool;

/// The in-process restart (audit RA-006 / RA-008 / RA-021): once the run stopped, its slot is
/// empty, nothing of the queue keeps the run's `AppState` alive (not even a request still
/// holding the stopped queue), a late request of that run gets a closed queue instead of a new
/// one, and the next run gets a fresh queue on its own state.
#[test]
fn a_stopped_run_leaves_no_queue_and_keeps_no_state_alive() {
    let control = super::super::test_util::test_state();
    let control_weak = Arc::downgrade(&control);
    drop(control);
    assert!(control_weak.upgrade().is_none(), "a test state alone frees");

    let slot = QueueSlot::new();
    let state = super::super::test_util::test_state();
    let weak = Arc::downgrade(&state);
    let shared = Shared::from(state.clone());
    let svc = slot.get_or_start(&shared);
    assert!(!svc.stopped());
    assert!(Arc::ptr_eq(&svc, &slot.get_or_start(&shared)), "one queue");
    assert!(slot.get().is_some_and(|s| Arc::ptr_eq(&s, &svc)));

    slot.stop_for_exit(std::time::Instant::now() + std::time::Duration::from_secs(10));
    assert!(slot.get().is_none(), "the slot is empty");
    assert!(svc.stopped());
    // A late request of the stopped run: a closed queue, never stored.
    let late = slot.get_or_start(&shared);
    assert!(late.stopped());
    assert!(slot.get().is_none());
    drop(late);
    drop((shared, state));
    assert!(
        weak.upgrade().is_none(),
        "the stopped queue (still held by a request) keeps no AppState"
    );
    drop(svc);

    // The next run of the process.
    let next = super::super::test_util::test_state();
    let next_shared = Shared::from(next.clone());
    let fresh = slot.get_or_start(&next_shared);
    assert!(!fresh.stopped());
    assert!(slot.get().is_some());
    slot.stop_for_exit(std::time::Instant::now() + std::time::Duration::from_secs(10));
}

/// A run whose stop request came (its `shutdown` token) never starts a queue.
#[test]
fn a_stopping_run_gets_a_closed_queue() {
    let slot = QueueSlot::new();
    let state = super::super::test_util::test_state();
    state.shutdown.cancel();
    let svc = slot.get_or_start(&Shared::from(state.clone()));
    assert!(svc.stopped());
    assert!(slot.get().is_none());
    let weak = Arc::downgrade(&state);
    drop(state);
    assert!(weak.upgrade().is_none());
    drop(svc);
}

/// The hooks follow a generation restart (the run's `Shared` swaps its state) and fail with a
/// message, never act on a dead run, once detached.
#[test]
fn the_target_follows_the_run_until_it_is_detached() {
    let first = super::super::test_util::test_state();
    let shared = Shared::from(first.clone());
    let target = Target::new(&shared);
    assert!(target.state().is_some_and(|s| Arc::ptr_eq(&s, &first)));
    let second = super::super::test_util::test_state();
    shared.swap(second.clone());
    assert!(target.state().is_some_and(|s| Arc::ptr_eq(&s, &second)));
    target.detach();
    assert!(target.state().is_none());
    let e = target
        .require("add m to the config")
        .unwrap_err()
        .to_string();
    assert!(
        e.starts_with("cannot add m to the config: the server is stopping"),
        "{e}"
    );
    let hooks = ServerHooks {
        target: target.clone(),
    };
    let out = TrainOutput {
        name: "m".into(),
        onnx: "m.onnx".into(),
        yaml: "m.yaml".into(),
        provenance: "m.provenance.json".into(),
        family: ModelFamilyKind::Yolo26,
        classes: vec!["person".into()],
        license: "test".into(),
        heldout: None,
        device: "cpu".into(),
        train_seconds: 1.0,
    };
    assert!(
        hooks.register(&out).is_err(),
        "nothing is saved for a stopped run"
    );
    let fetch = RunFetcher::new(target);
    let res = crate::resources::catalog::all().next().unwrap();
    let e = crate::resources::export::Fetcher::fetch(
        &fetch,
        res,
        Path::new("/nonexistent"),
        &mut |_, _| {},
        &AtomicBool::new(false),
    )
    .unwrap_err();
    assert!(e.to_string().contains("the server is stopping"), "{e}");
}

#[test]
fn the_preview_counts_unreviewed_frames_from_the_strips_walk_and_never_walks_itself() {
    use crate::collect::labels::bulk::test_support::put_frame;
    use crate::collect::labels::{self, Frame, LabelFile, Status};
    let root = std::env::temp_dir().join(format!("bop-unrev-{}", uuid::Uuid::new_v4()));
    let dir = root.join("review");
    let mut config = Config::default();
    config.collect.review_dir = Some(dir.clone());
    // No folder: not known, and no walk was started for it.
    assert_eq!(unreviewed_now(&config), None);
    put_frame(&dir, "open", None, vec![]);
    let done = put_frame(&dir, "done", None, vec![]);
    let reviewed = LabelFile {
        status: Status::Reviewed,
        ..LabelFile::default()
    };
    labels::write(&Frame::resolve(&dir, &done).unwrap().labels, &reviewed).unwrap();
    // The first answer may take the walk's own time on a busy machine; it is the same number.
    let walked = super::super::train_area::tally_of(&dir, std::time::Duration::from_secs(60));
    assert_eq!(walked.unwrap().to_review, 1);
    assert_eq!(
        unreviewed_now(&config),
        crate::training::unreviewed_frames(&config)
    );
    let _ = std::fs::remove_dir_all(&root);
}

fn form(pairs: &[(&str, &str)]) -> Vec<(String, String)> {
    pairs
        .iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}

fn mac() -> Capability {
    crate::training::capability::capability(
        &HardwareInfo::new("macos", "aarch64", vec![apple_gpu("Apple M1")]),
        Some(16.0),
    )
}

/// The queue form's "Start anyway" reaches every request it makes (each seed is checked at its
/// own start); without it, none.
#[test]
fn the_queue_form_carries_start_anyway() {
    let cap = mac();
    let training = crate::training::config::TrainingConfig::default();
    let base = [("model", "yolo26n"), ("dataset", "/x"), ("seeds", "2")];
    let (reqs, _) = requests_from_form(&form(&base), &cap, &training).unwrap();
    assert!(!reqs.is_empty() && reqs.iter().all(|r| !r.ignore_memory_check));
    let mut on = base.to_vec();
    on.push(("ignore_memory_check", "1"));
    let (reqs, _) = requests_from_form(&form(&on), &cap, &training).unwrap();
    assert_eq!(reqs.len(), 2);
    assert!(reqs.iter().all(|r| r.ignore_memory_check));
}

/// QA-094: the queue form reads a false-like checkbox as unticked (`confirm_large=0` allowed
/// the multi-GB training environment before).
#[test]
fn the_queue_form_reads_false_like_checkboxes_as_unticked() {
    let cap = mac();
    let training = crate::training::config::TrainingConfig::default();
    for v in crate::server::form::FALSE_LIKE {
        let f = form(&[
            ("model", "yolo26n"),
            ("dataset", "/x"),
            ("confirm_large", v),
            ("ignore_memory_check", v),
            ("public", v),
        ]);
        let (reqs, _) = requests_from_form(&f, &cap, &training).unwrap();
        assert!(!reqs.is_empty(), "{v:?}");
        for r in reqs {
            assert!(!r.allow_large, "{v:?}");
            assert!(!r.ignore_memory_check && !r.include_public, "{v:?}");
        }
    }
    let f = form(&[
        ("model", "yolo26n"),
        ("dataset", "/x"),
        ("confirm_large", "on"),
    ]);
    let (reqs, _) = requests_from_form(&f, &cap, &training).unwrap();
    assert!(reqs.iter().all(|r| r.allow_large));
}

/// The form and `train run` build their recipe through `recipe::build_recipe`: the same
/// typed values give the same recipe whichever edge parsed them.
#[test]
fn the_form_recipe_is_the_shared_builder_s() {
    let cap = mac();
    let training = crate::training::config::TrainingConfig::default();
    let f = form(&[
        ("model", "yolo26n"),
        ("dataset", "/x"),
        ("epochs", "12"),
        ("patience", "4"),
        ("batch", "3"),
        ("imgsz", "480"),
        ("lr", "0.002"),
        ("lr_schedule", "cosine"),
        ("lr_min_factor", "0.05"),
        ("night_aug", "1"),
        ("multi_scale", "off"),
    ]);
    let (reqs, _) = requests_from_form(&f, &cap, &training).unwrap();
    let direct = recipe::build_recipe(
        crate::resources::catalog::TrainFamily::Yolo26,
        "n",
        &recipe::RecipeInputs {
            epochs: Some(12),
            patience: Some(4),
            imgsz: Some(480),
            batch: Some(3),
            lr: Some(0.002),
            lr_schedule: Some(recipe::LrSchedule::Cosine),
            lr_min_factor: Some(0.05),
            select_metric: None,
            augment: Augment::NIGHT,
            multi_scale: recipe::MultiScale::parse("off").unwrap(),
        },
        &training,
        &cap,
    )
    .unwrap();
    assert_eq!(reqs[0].recipe, direct);
}

#[test]
fn the_queue_form_takes_a_gpu_and_suggests_its_batch() {
    use crate::backend::detect::{ComputeCapability, CudaInfo, GpuAdapter, GpuVendor};
    let gpu = |name: &str, vram_mb: u64, ordinal: u32| GpuAdapter {
        vendor: GpuVendor::Nvidia,
        name: name.into(),
        vram_mb,
        index: ordinal,
        discrete: true,
        cuda: Some(CudaInfo {
            ordinal,
            compute_capability: ComputeCapability::new(8, 6),
        }),
    };
    let cap = crate::training::capability::capability(
        &HardwareInfo::new(
            "linux",
            "x86_64",
            vec![gpu("RTX 3090", 24576, 0), gpu("RTX 3050", 4096, 1)],
        )
        .with_cuda_driver(12, 8),
        None,
    );
    let post = |g: &str| {
        requests_from_form(
            &form(&[("model", "rfdetr-medium"), ("dataset", "/x"), ("gpu", g)]),
            &cap,
            &Default::default(),
        )
        .map(|(r, _)| r.into_iter().next().unwrap())
    };
    let auto = post("").unwrap();
    assert_eq!(auto.gpu, None);
    let small = post("cuda:1").unwrap();
    assert_eq!(small.gpu.as_deref(), Some("cuda:1"));
    assert!(
        small.recipe.batch < auto.recipe.batch,
        "4 GB suggests a smaller batch: {} vs {}",
        small.recipe.batch,
        auto.recipe.batch
    );
    let e = post("cuda:5").unwrap_err().to_string();
    assert!(
        e.contains("no such CUDA device") && e.contains("RTX 3050"),
        "{e}"
    );
    let e = requests_from_form(
        &form(&[("model", "yolo26n"), ("dataset", "/x"), ("gpu", "cuda:0")]),
        &mac(),
        &Default::default(),
    )
    .unwrap_err()
    .to_string();
    assert!(e.contains("no GPU to choose"), "{e}");
}

#[test]
fn the_queue_form_takes_benchmark_devices_else_the_setting() {
    let training = crate::training::config::TrainingConfig {
        benchmark_devices: vec!["ort:cuda:0".into()],
        ..Default::default()
    };
    // The fieldset sent: its ticks, canonical and deduplicated, in order.
    let f = form(&[
        ("model", "yolo26n"),
        ("dataset", "/x"),
        ("benchmark_devices_form", "1"),
        ("benchmark_device", "GPU"),
        ("benchmark_device", "ort:cpu"),
        ("benchmark_device", "openvino:gpu"),
    ]);
    let (reqs, _) = requests_from_form(&f, &mac(), &training).unwrap();
    assert_eq!(reqs[0].benchmark_devices, ["openvino:gpu", "ort:cpu"]);
    // The fieldset with nothing ticked: the device the model would load on.
    let f = form(&[
        ("model", "yolo26n"),
        ("dataset", "/x"),
        ("benchmark_devices_form", "1"),
    ]);
    let (reqs, _) = requests_from_form(&f, &mac(), &training).unwrap();
    assert!(reqs[0].benchmark_devices.is_empty());
    // No fieldset (a script): the saved setting.
    let f = form(&[("model", "yolo26n"), ("dataset", "/x")]);
    let (reqs, _) = requests_from_form(&f, &mac(), &training).unwrap();
    assert_eq!(reqs[0].benchmark_devices, ["ort:cuda:0"]);
    // `auto` is not one device.
    let f = form(&[
        ("model", "yolo26n"),
        ("dataset", "/x"),
        ("benchmark_devices_form", "1"),
        ("benchmark_device", "auto"),
    ]);
    let e = requests_from_form(&f, &mac(), &training)
        .unwrap_err()
        .to_string();
    assert!(e.contains("benchmark devices"), "{e}");
}

#[test]
fn seeds_repeat_every_run_in_one_group() {
    let f = form(&[
        ("model", "yolo26n"),
        ("dataset", "/x"),
        ("scene", "day"),
        ("scene", "night+ir"),
        ("seeds", "3"),
    ]);
    let (reqs, name) = requests_from_form(&f, &mac(), &Default::default()).unwrap();
    assert_eq!(name, None);
    assert_eq!(reqs.len(), 6);
    let seeds: Vec<Option<u64>> = reqs.iter().map(|r| r.train_seed).collect();
    assert_eq!(
        seeds,
        [Some(0), Some(1), Some(2), Some(0), Some(1), Some(2)]
    );
    let group = reqs[0].compare_group.clone();
    assert!(group.is_some() && reqs.iter().all(|r| r.compare_group == group));
    // One scene, one model, three seeds: still a group; a typed name is allowed.
    let f = form(&[
        ("model", "yolo26n"),
        ("dataset", "/x"),
        ("seeds", "3"),
        ("name", "cam"),
    ]);
    let (reqs, name) = requests_from_form(&f, &mac(), &Default::default()).unwrap();
    assert_eq!((reqs.len(), name.as_deref()), (3, Some("cam")));
    assert!(reqs[0].compare_group.is_some());
    // One seed (the default): no train seed, no group.
    for extra in [&[][..], &[("seeds", "1")][..]] {
        let mut v = vec![("model", "yolo26n"), ("dataset", "/x")];
        v.extend_from_slice(extra);
        let (reqs, _) = requests_from_form(&form(&v), &mac(), &Default::default()).unwrap();
        assert_eq!(reqs.len(), 1);
        assert_eq!(reqs[0].train_seed, None);
        assert_eq!(reqs[0].compare_group, None);
    }
    for bad in ["0", "4", "x"] {
        let f = form(&[("model", "yolo26n"), ("dataset", "/x"), ("seeds", bad)]);
        let e = requests_from_form(&f, &mac(), &Default::default())
            .unwrap_err()
            .to_string();
        assert!(e.contains("seeds"), "{e}");
    }
}

#[test]
fn the_queue_form_makes_one_request_per_model() {
    let f = form(&[
        ("model", "yolo26n"),
        ("model", "rfdetr-nano"),
        ("dataset", "/exports/a"),
        ("dataset_path", "/exports/b"),
        ("scene", "night+ir"),
        ("classes", "cat, raccoon"),
        ("night_aug", "on"),
        ("aug_noise", "0.1"),
        ("epochs", "3"),
        ("benchmark_set", "nir-night"),
        ("benchmark_set", "evil"),
    ]);
    let (reqs, name) = requests_from_form(&f, &mac(), &Default::default()).unwrap();
    assert_eq!(name, None);
    assert_eq!(reqs.len(), 2);
    let r = &reqs[1];
    assert_eq!(r.recipe.label(), "rfdetr-nano");
    assert_eq!(r.recipe.epochs, 3);
    assert_eq!(
        r.recipe.augment,
        Augment {
            dark: 0.5,
            noise: 0.1,
            gray: 0.5
        }
    );
    assert_eq!(
        r.datasets,
        [PathBuf::from("/exports/a"), PathBuf::from("/exports/b")]
    );
    assert_eq!(r.scene, SceneFilter::NIGHT_IR);
    assert_eq!(
        r.classes.as_deref(),
        Some(&["cat".to_string(), "raccoon".to_string()][..])
    );
    assert_eq!(r.benchmark_sets, ["nir-night"]);
    assert!(r.recipe.batch >= 1 && r.recipe.batch <= 16);
    assert!(!r.include_public && !r.allow_large);
    // The download confirmation also allows the public night subset.
    let (reqs, _) = requests_from_form(
        &form(&[
            ("model", "yolo26n"),
            ("dataset", "/x"),
            ("public", "on"),
            ("confirm_large", "on"),
        ]),
        &mac(),
        &Default::default(),
    )
    .unwrap();
    assert!(reqs[0].include_public && reqs[0].allow_large);
    assert_eq!(reqs[0].install_dir, None);
    let (reqs, _) = requests_from_form(
        &form(&[("model", "yolo26n"), ("install_dir", " /srv/models ")]),
        &mac(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(reqs[0].install_dir, Some(PathBuf::from("/srv/models")));
    let e =
        requests_from_form(&form(&[("dataset", "/x")]), &mac(), &Default::default()).unwrap_err();
    assert!(e.to_string().contains("no model ticked"));
    let e = requests_from_form(
        &form(&[("model", "yolo26n"), ("epochs", "lots")]),
        &mac(),
        &Default::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("epochs"));
    let e = requests_from_form(
        &form(&[("model", "yolo26n"), ("model", "yolo26s"), ("name", "x")]),
        &mac(),
        &Default::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("one ticked model"));
}

#[test]
fn form_epochs_and_patience_fall_back_to_the_config_then_the_family() {
    let post = |extra: &[(&str, &str)], t: &crate::training::config::TrainingConfig| {
        let mut f = vec![("model", "yolo26n"), ("model", "dfine-s")];
        f.extend_from_slice(extra);
        let (reqs, _) = requests_from_form(&form(&f), &mac(), t).unwrap();
        reqs.iter()
            .map(|r| (r.recipe.epochs, r.recipe.patience))
            .collect::<Vec<_>>()
    };
    let none = crate::training::config::TrainingConfig::default();
    assert_eq!(post(&[], &none), [(100, 15), (80, 10)]);
    assert_eq!(
        post(&[("epochs", ""), ("patience", "")], &none),
        [(100, 15), (80, 10)]
    );
    let set = crate::training::config::TrainingConfig {
        epochs: Some(12),
        patience: Some(4),
        ..none.clone()
    };
    assert_eq!(post(&[], &set), [(12, 4), (12, 4)]);
    assert_eq!(
        post(&[("epochs", "6"), ("patience", "0")], &set),
        [(6, 0), (6, 0)]
    );
    // A recommended patience above typed epochs is capped by them.
    assert_eq!(post(&[("epochs", "5")], &none), [(5, 5), (5, 5)]);
}

#[test]
fn form_learning_rate_settings_fall_back_to_the_config_then_the_family() {
    use crate::training::recipe::LrSchedule;
    type Lr = (Option<f32>, Option<LrSchedule>, Option<f32>);
    let post = |models: &[&str],
                extra: &[(&str, &str)],
                t: &crate::training::config::TrainingConfig|
     -> anyhow::Result<Vec<Lr>> {
        let mut f: Vec<(&str, &str)> = models.iter().map(|m| ("model", *m)).collect();
        f.extend_from_slice(extra);
        let (reqs, _) = requests_from_form(&form(&f), &mac(), t)?;
        Ok(reqs
            .iter()
            .map(|r| (r.recipe.lr, r.recipe.lr_schedule, r.recipe.lr_min_factor))
            .collect())
    };
    let none = crate::training::config::TrainingConfig::default();
    let both = ["yolo26n", "dfine-s"];
    assert_eq!(
        post(&both, &[], &none).unwrap(),
        [(None, None, None), (None, None, None)]
    );
    assert_eq!(
        post(
            &both,
            &[
                ("lr", ""),
                ("lr_schedule", "default"),
                ("lr_min_factor", "")
            ],
            &none
        )
        .unwrap(),
        [(None, None, None), (None, None, None)]
    );
    let typed = [
        ("lr", "0.002"),
        ("lr_schedule", "cosine"),
        ("lr_min_factor", "0.05"),
    ];
    let want = (Some(0.002), Some(LrSchedule::Cosine), Some(0.05));
    assert_eq!(post(&both, &typed, &none).unwrap(), [want, want]);
    // The config fills blank fields; a stored schedule a family does not take is its default.
    let set = crate::training::config::TrainingConfig {
        lr: Some(1e-4),
        lr_schedule: Some(LrSchedule::Step),
        lr_min_factor: Some(0.2),
        ..none.clone()
    };
    assert_eq!(
        post(&["rfdetr-nano", "yolo26n"], &[], &set).unwrap(),
        [
            (Some(1e-4), Some(LrSchedule::Step), Some(0.2)),
            (Some(1e-4), None, Some(0.2))
        ]
    );
    assert_eq!(post(&both, &typed, &set).unwrap(), [want, want]);
    // Bad values name the field.
    let e = post(&both, &[("lr_schedule", "exp")], &none).unwrap_err();
    assert!(e.to_string().contains("LR schedule"), "{e}");
    let e = post(&both, &[("lr_min_factor", "lots")], &none).unwrap_err();
    assert!(e.to_string().contains("final LR factor"), "{e}");
}

#[test]
fn the_selected_metric_comes_from_the_form_else_the_config() {
    use crate::training::config::TrainingConfig;
    use crate::training::recipe::SelectMetric;
    type Queued = (SelectMetric, Option<u64>, Option<String>);
    let post = |extra: &[(&str, &str)], t: &TrainingConfig| -> anyhow::Result<Vec<Queued>> {
        let mut f: Vec<(&str, &str)> = vec![("model", "yolo26n"), ("dataset", "/x")];
        f.extend_from_slice(extra);
        let (reqs, _) = requests_from_form(&form(&f), &mac(), t)?;
        Ok(reqs
            .into_iter()
            .map(|r| (r.recipe.select_metric, r.train_seed, r.compare_group))
            .collect())
    };
    let none = TrainingConfig::default();
    let metric = |v: &[Queued]| v[0].0;
    assert_eq!(metric(&post(&[], &none).unwrap()), SelectMetric::Map);
    assert_eq!(
        metric(&post(&[("select_metric", "")], &none).unwrap()),
        SelectMetric::Map
    );
    assert_eq!(
        metric(&post(&[("select_metric", "f0_5")], &none).unwrap()),
        SelectMetric::F05
    );
    // The form wins over the config, also with the default.
    let set = TrainingConfig {
        select_metric: Some(SelectMetric::Iou),
        ..none.clone()
    };
    assert_eq!(metric(&post(&[], &set).unwrap()), SelectMetric::Iou);
    assert_eq!(
        metric(&post(&[("select_metric", "map")], &set).unwrap()),
        SelectMetric::Map
    );
    // Every seed and scene of a compare group carries it.
    let group = post(
        &[
            ("select_metric", "f2"),
            ("seeds", "2"),
            ("scene", "day"),
            ("scene", "night"),
        ],
        &none,
    )
    .unwrap();
    assert_eq!(group.len(), 4);
    assert!(group.iter().all(|r| r.0 == SelectMetric::F2));
    assert!(group.iter().all(|r| r.2.is_some() && r.2 == group[0].2));
    let e = post(&[("select_metric", "f3")], &none)
        .unwrap_err()
        .to_string();
    assert!(e.contains("select best by") && e.contains("f0_5"), "{e}");
}

#[test]
fn dataset_folders_are_offered_or_allowed() {
    let offered = [PathBuf::from("/exports/a"), PathBuf::from("exports/rel")];
    let allow = |raw: &str| -> Result<PathBuf, String> {
        if raw.starts_with("/ok/") {
            Ok(PathBuf::from(format!("/canon{raw}")))
        } else {
            Err(format!("{raw} is outside the browsable folders"))
        }
    };
    // Offered ones as they are (a relative saved entry too), others through the check.
    let f = form(&[
        ("dataset", "/exports/a"),
        ("dataset", "exports/rel"),
        ("dataset", "/ok/b"),
        ("dataset_path", "/ok/c"),
    ]);
    assert_eq!(
        checked_datasets(&f, &offered, &allow).unwrap(),
        [
            PathBuf::from("/exports/a"),
            PathBuf::from("exports/rel"),
            PathBuf::from("/canon/ok/b"),
            PathBuf::from("/canon/ok/c")
        ]
    );
    // A ticked value the page did not offer.
    let e = checked_datasets(&form(&[("dataset", "/etc/x")]), &offered, &allow).unwrap_err();
    assert!(
        e.to_string().starts_with("datasets: /etc/x is outside"),
        "{e}"
    );
    // The typed folder is always checked, even when it is an offered one.
    let e =
        checked_datasets(&form(&[("dataset_path", "/exports/a")]), &offered, &allow).unwrap_err();
    assert!(e.to_string().starts_with("another folder:"), "{e}");
    assert!(
        checked_datasets(&form(&[("dataset_path", " ")]), &offered, &allow)
            .unwrap()
            .is_empty()
    );
}

#[test]
fn folder_settings_are_checked() {
    let dir = std::env::temp_dir().join(format!("bop-train-folder-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let file = dir.join("f.txt");
    std::fs::write(&file, b"x").unwrap();
    let allow = |raw: &str| -> Result<PathBuf, String> {
        std::path::Path::new(raw)
            .canonicalize()
            .map_err(|_| format!("{raw} does not exist"))
    };
    let root = dir.as_path();
    assert_eq!(checked_folder("x", None, root, &allow).unwrap(), None);
    assert_eq!(checked_folder("x", Some("  "), root, &allow).unwrap(), None);
    assert_eq!(
        checked_folder("x", Some(&dir.display().to_string()), root, &allow).unwrap(),
        Some(dir.canonicalize().unwrap())
    );
    // Relative: under the data root.
    std::fs::create_dir_all(dir.join("sub")).unwrap();
    assert_eq!(
        checked_folder("x", Some("sub"), root, &allow).unwrap(),
        Some(dir.join("sub").canonicalize().unwrap())
    );
    let e = checked_folder("install the model in", Some("nope"), root, &allow).unwrap_err();
    assert!(e.to_string().starts_with("install the model in: "), "{e}");
    let e = checked_folder("x", Some(&file.display().to_string()), root, &allow).unwrap_err();
    assert!(e.to_string().contains("is not a folder"), "{e}");
    let e = checked_folder("training folder", Some("/usr/lib"), root, &allow).unwrap_err();
    assert!(e.to_string().contains("system directory"), "{e}");
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn the_datasets_form_reads_the_queue_forms_buttons() {
    // A saved folder's Remove button inside the queue form: only that folder.
    let f = form(&[
        ("dataset", "/exports/a"),
        ("dataset_path", "/exports/new"),
        ("model", "yolo26n"),
        ("epochs", "50"),
        ("remove", "/exports/a"),
    ]);
    assert_eq!(
        dataset_edit(&f).unwrap(),
        DatasetEdit::Remove(PathBuf::from("/exports/a"))
    );
    // Save folder: the "Another folder" field.
    let f = form(&[
        ("dataset", "/exports/a"),
        ("dataset_path", " /exports/new "),
    ]);
    assert_eq!(
        dataset_edit(&f).unwrap(),
        DatasetEdit::Add(PathBuf::from("/exports/new"))
    );
    // The older form's `add`.
    assert_eq!(
        dataset_edit(&form(&[("add", "/x")])).unwrap(),
        DatasetEdit::Add(PathBuf::from("/x"))
    );
    let e = dataset_edit(&form(&[("dataset_path", ""), ("model", "yolo26n")])).unwrap_err();
    assert!(e.to_string().contains("another folder"), "{e}");
}

#[test]
fn latency_hints_split_into_class_and_text() {
    assert_eq!(
        latency_parts("light: runs anywhere"),
        ("light", "runs anywhere")
    );
    assert_eq!(latency_parts("odd"), ("", "odd"));
}

#[test]
fn use_as_writes_the_default_night_and_verify_models() {
    let dir = std::env::temp_dir().join(format!("bop-train-use-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let onnx = dir.join("night-x.onnx");
    std::fs::write(&onnx, vec![1u8; 2048]).unwrap();
    let yaml = dir.join("night-x.yaml");
    std::fs::write(&yaml, "NAMES:\n  - cat\n").unwrap();
    let out = TrainOutput {
        name: "night-x".into(),
        onnx,
        yaml,
        provenance: dir.join("night-x.provenance.json"),
        family: ModelFamilyKind::Yolo26,
        classes: vec!["cat".into()],
        license: "AGPL-3.0".into(),
        heldout: None,
        device: "mps".into(),
        train_seconds: 1.0,
    };
    let mut c = Config::default();
    c.seed_default_model();
    let opts = UseOptions::default();
    apply_use(&mut c, &out, "night", None, &opts).unwrap();
    assert_eq!(c.models[0].night_model.as_deref(), Some("night-x"));
    assert!(
        c.models
            .iter()
            .any(|m| m.effective_name() == "night-x" && m.enabled)
    );
    apply_use(&mut c, &out, "verify", Some("IPcam-general"), &opts).unwrap();
    assert_eq!(c.models[0].verify.as_ref().unwrap().model, "night-x");
    apply_use(&mut c, &out, "default", None, &opts).unwrap();
    assert_eq!(c.default_model.as_deref(), Some("night-x"));
    assert!(apply_use(&mut c, &out, "night", Some("night-x"), &opts).is_err());
    assert!(apply_use(&mut c, &out, "boss", None, &opts).is_err());
    assert!(apply_use(&mut c, &out, "night", Some("nope"), &opts).is_err());
    let _ = std::fs::remove_dir_all(&dir);
}

#[test]
fn use_as_verifier_asks_before_replacing_and_default_refuses_a_stale_page() {
    let dir = std::env::temp_dir().join(format!("bop-train-use-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&dir).unwrap();
    let onnx = dir.join("new-x.onnx");
    std::fs::write(&onnx, vec![1u8; 2048]).unwrap();
    let yaml = dir.join("new-x.yaml");
    std::fs::write(&yaml, "NAMES:\n  - person\n").unwrap();
    let out = TrainOutput {
        name: "new-x".into(),
        onnx,
        yaml,
        provenance: dir.join("new-x.provenance.json"),
        family: ModelFamilyKind::Yolo26,
        classes: vec!["person".into()],
        license: "AGPL-3.0".into(),
        heldout: None,
        device: "mps".into(),
        train_seconds: 1.0,
    };
    let mut c = Config::default();
    c.seed_default_model();
    c.models.push(crate::config::ModelConfig {
        name: Some("checker".into()),
        path: "models/checker.onnx".into(),
        ..Default::default()
    });
    c.models[0].verify = Some(crate::verify::VerifyConfig::with_model("checker"));
    let default = UseOptions::default();
    let e = apply_use(&mut c, &out, "verify", Some("IPcam-general"), &default).unwrap_err();
    assert!(e.to_string().contains("checker already verifies"), "{e}");
    assert_eq!(c.models[0].verify.as_ref().unwrap().model, "checker");
    let yes = UseOptions {
        replace_verifier: Some("checker".into()),
        ..Default::default()
    };
    apply_use(&mut c, &out, "verify", Some("IPcam-general"), &yes).unwrap();
    assert_eq!(c.models[0].verify.as_ref().unwrap().model, "new-x");
    // The preview showed another default: nothing is switched on stale facts.
    let stale = UseOptions {
        expect_current: Some("checker".into()),
        ..Default::default()
    };
    let e = apply_use(&mut c, &out, "default", None, &stale).unwrap_err();
    assert!(e.to_string().contains("changed to IPcam-general"), "{e}");
    assert_eq!(c.default_model, None);
    let msg = apply_use(&mut c, &out, "default", None, &UseOptions::default()).unwrap();
    assert!(msg.contains("new-x is now the default model"), "{msg}");
    let p = c.previous_default.as_ref().unwrap();
    assert_eq!(p.model, "IPcam-general");
    let _ = std::fs::remove_dir_all(&dir);
}

#[tokio::test]
async fn the_home_and_training_pages_offer_revert_and_revert_applies() {
    use crate::config::PreviousDefault;
    use crate::server::test_util::{call, get_text, post_form, test_state_with};
    let _queue = QUEUE_TEST_LOCK.lock().await;
    let root = std::env::temp_dir().join(format!("bop-train-rev-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| {
        c.download_dir = Some(r2);
        c.default_model = Some("<b>evil</b>".into());
        c.previous_default = Some(PreviousDefault {
            model: "ipcam-general".into(),
            default_model: Some("ipcam-general".into()),
            switched_to: "<b>evil</b>".into(),
            at: "2026-11-05T10:00:00Z".into(),
            ..Default::default()
        });
    });
    for page in ["/", "/train/models"] {
        let (_, html) = get_text(&state, page).await;
        assert!(html.contains("Revert to ipcam-general"), "{page}");
        assert!(html.contains("action=\"/training/revert\""), "{page}");
        assert!(
            !html.contains("switched from <code>ipcam-general</code> to <code><b>"),
            "{page}"
        );
    }
    let (_, _, b) = call(&state, post_form("/training/revert", "", true)).await;
    let html = String::from_utf8(b).unwrap();
    assert!(html.contains("Back to ipcam-general"), "{html}");
    assert_eq!(
        state.config_read().default_model.as_deref(),
        Some("ipcam-general")
    );
    let (_, html) = get_text(&state, "/").await;
    assert!(
        html.contains("Revert to &#60;b&#62;evil") || html.contains("Revert to &lt;b&gt;evil"),
        "the undo of the undo"
    );
    // A record that cannot be used can be dropped.
    let (_, _, _) = call(&state, post_form("/training/revert", "forget=1", true)).await;
    assert_eq!(state.config_read().previous_default, None);
    let (_, _, b) = call(&state, post_form("/training/revert", "", true)).await;
    assert!(
        String::from_utf8(b)
            .unwrap()
            .contains("Not reverted: no default switch")
    );
    // The preview of an unknown run says what to do.
    let (s, text) = get_text(&state, "/v1/training/switch-preview?id=nope").await;
    assert_eq!(s, StatusCode::NOT_FOUND);
    assert!(text.contains("no run nope"), "{text}");
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn the_poll_answer_is_the_light_one_only_when_asked_and_jobs_carry_tasks() {
    use crate::server::test_util::{get_text, test_state_with};
    let _queue = QUEUE_TEST_LOCK.lock().await;
    let root = std::env::temp_dir().join(format!("bop-train-poll-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let get = |path: &'static str| {
        let state = state.clone();
        async move {
            let (s, text) = get_text(&state, path).await;
            assert_eq!(s, StatusCode::OK, "{path}: {text}");
            serde_json::from_str::<serde_json::Value>(&text).unwrap()
        }
    };
    let full = get("/v1/training").await;
    assert!(full.get("capability").is_some() && full.get("metrics").is_some());
    let first = get("/v1/training?poll=1").await;
    assert!(first.get("capability").is_none(), "{first}");
    assert!(first.get("metrics").is_some() && first.get("runs").is_some());
    let next = get("/v1/training?poll=1&have=").await;
    assert!(next.get("metrics").is_none() && next.get("tasks").is_some());
    // poll=0 is the full answer.
    assert!(get("/v1/training?poll=0").await.get("capability").is_some());
    let jobs = get("/v1/train/jobs").await;
    assert!(jobs.get("task").is_none() && jobs["jobs"].get("active").is_some());
    let jobs = get("/v1/train/jobs?task=nope&tasks=busy").await;
    assert!(jobs["task"].is_null() && jobs["tasks"].is_array(), "{jobs}");
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn the_page_and_json_render_and_bad_forms_say_why() {
    use crate::server::test_util::{call, get_text, post_form, test_state_with};
    let _queue = QUEUE_TEST_LOCK.lock().await;
    let root = std::env::temp_dir().join(format!("bop-train-page-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let (s, page) = get_text(&state, "/train/train").await;
    assert_eq!(s, StatusCode::OK);
    assert!(
        page.contains("href=\"/train\" class=\"active\" aria-current=\"page\">Train</a>"),
        "nav"
    );
    assert!(page.contains("This machine"));
    assert!(page.contains("/static/train-train.js"));
    let (s, json) = get_text(&state, "/v1/training").await;
    assert_eq!(s, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    assert!(
        v["capability"]["families"].as_array().unwrap().len() == 6,
        "{v}"
    );
    assert!(v["runs"].as_array().unwrap().is_empty());
    // No model ticked: refused before anything is queued.
    let (_, _, b) = call(&state, post_form("/training/queue", "dataset=%2Fx", true)).await;
    let html = String::from_utf8(b).unwrap();
    assert!(
        html.contains("Not queued") || html.contains("not available"),
        "{html}"
    );
    let (_, _, b) = call(
        &state,
        post_form("/training/use", "id=nope&role=night", true),
    )
    .await;
    assert!(String::from_utf8(b).unwrap().contains("no run nope"));
    let (_, _, b) = call(
        &state,
        post_form("/training/datasets", "add=%2Fno%2Fsuch%2Fdir", true),
    )
    .await;
    assert!(String::from_utf8(b).unwrap().contains("Not saved"));
    // Save folder from the queue form, with its other fields: the same check.
    let (_, _, b) = call(
        &state,
        post_form(
            "/training/datasets",
            "dataset_path=%2Fno%2Fsuch%2Fdir&model=yolo26n&epochs=50&save=1",
            true,
        ),
    )
    .await;
    assert!(String::from_utf8(b).unwrap().contains("Not saved"));
    assert!(state.config_read().training.datasets.is_empty());
    // A real export folder is saved, then removed by its button.
    let export = root.join("exports/review-1");
    std::fs::create_dir_all(&export).unwrap();
    std::fs::write(export.join("classes.txt"), "person\n").unwrap();
    let enc = super::super::encode_uri_component(&export.display().to_string());
    let (_, _, b) = call(
        &state,
        post_form(
            "/training/datasets",
            &format!("dataset_path={enc}&model=yolo26n"),
            true,
        ),
    )
    .await;
    let html = String::from_utf8(b).unwrap();
    assert!(!html.contains("Not saved"), "{html}");
    assert_eq!(
        state.config_read().training.datasets,
        [super::super::browse::plain(&export.canonicalize().unwrap())]
    );
    let enc = super::super::encode_uri_component(
        &super::super::browse::plain(&export.canonicalize().unwrap())
            .display()
            .to_string(),
    );
    assert!(html.contains("formaction=\"/training/datasets\" formnovalidate name=\"remove\""));
    let (_, _, _) = call(
        &state,
        post_form(
            "/training/datasets",
            &format!("dataset={enc}&dataset_path=&remove={enc}"),
            true,
        ),
    )
    .await;
    assert!(state.config_read().training.datasets.is_empty());
    // Queue: a dataset the page did not offer, outside the browsable folders, is refused
    // and named; so is an install folder outside them.
    let (_, _, b) = call(
        &state,
        post_form(
            "/training/queue",
            "model=yolo26n&dataset=%2Fno%2Fsuch%2Fexport",
            true,
        ),
    )
    .await;
    let html = String::from_utf8(b).unwrap();
    // Refused either way: outside the roots, or (Windows: drive-relative, so under the
    // data root) not found.
    assert!(
        html.contains("Not queued: datasets: ") && html.contains("/no/such/export"),
        "{html}"
    );
    let (_, _, b) = call(
        &state,
        post_form(
            "/training/queue",
            &format!("model=yolo26n&dataset={enc}&install_dir=%2Fno%2Fsuch%2Fdir"),
            true,
        ),
    )
    .await;
    let html = String::from_utf8(b).unwrap();
    assert!(
        html.contains("Not queued: install the model in: "),
        "{html}"
    );
    // The training folder: shown, set to a folder under the data root, back to the default
    // (once the process's queue is idle: another test's job may still run).
    assert!(page.contains("action=\"/training/dir\""));
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
    while started_queue().is_some_and(|s| s.busy()) {
        assert!(std::time::Instant::now() < deadline, "the queue stays busy");
        tokio::time::sleep(std::time::Duration::from_millis(50)).await;
    }
    let runs = root.join("my-runs");
    std::fs::create_dir_all(&runs).unwrap();
    let (s, _, b) = call(
        &state,
        post_form("/training/dir", "training_dir=my-runs", true),
    )
    .await;
    let html = String::from_utf8(b).unwrap();
    assert_eq!(s, StatusCode::OK);
    assert!(html.contains("stay there (not moved)"), "{html}");
    let runs = super::super::browse::plain(&runs.canonicalize().unwrap());
    assert_eq!(state.config_read().training.dir, Some(runs.clone()));
    assert_eq!(crate::training::training_dir(&state.config_read()), runs);
    let (_, _, b) = call(
        &state,
        post_form("/training/dir", "training_dir=%2Fno%2Fsuch%2Fdir", true),
    )
    .await;
    assert!(
        String::from_utf8(b)
            .unwrap()
            .contains("Not saved: training folder: ")
    );
    assert!(state.config_read().training.dir.is_some());
    let (_, _, _) = call(&state, post_form("/training/dir", "training_dir=", true)).await;
    assert_eq!(state.config_read().training.dir, None);
    let _ = std::fs::remove_dir_all(&root);
}

/// A running job shows Pause and Cancel, a paused one Resume and Cancel; the JSON says
/// `paused`; pausing a job that is not on the queue's thread is refused.
#[tokio::test]
async fn metric_columns_are_listed_saved_and_checked() {
    use crate::server::test_util::{REMOTE, call, from_peer, get_text, test_state_with};
    use axum::body::Body;
    use axum::http::{Request, header};
    let root = std::env::temp_dir().join(format!("bop-train-cols-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let post = |body: &str| {
        Request::post("/v1/training/metric-columns")
            .header(header::CONTENT_TYPE, "application/json")
            .body(Body::from(body.to_string()))
            .unwrap()
    };
    let json = |b: Vec<u8>| -> serde_json::Value { serde_json::from_slice(&b).unwrap() };
    let (s, text) = get_text(&state, "/v1/training").await;
    assert_eq!(s, StatusCode::OK);
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        v["metric_columns"],
        serde_json::json!([
            "map",
            "map50",
            "iou",
            "f1",
            "f2",
            "precision",
            "recall",
            "loss"
        ])
    );
    let metrics = v["metrics"].as_array().unwrap();
    assert_eq!(metrics.len(), crate::training::progress::METRICS.len());
    let map = metrics.iter().find(|m| m["id"] == "map").unwrap();
    assert_eq!(map["label"], "mAP 50-95");
    assert_eq!(map["short"], "mAP");
    assert_eq!(map["higher_better"], true);
    assert_eq!(map["format"], "fixed3");

    // Saved (trimmed), raw names allowed, answered and served.
    let (s, _, b) = call(
        &state,
        post(r#"{"columns": [" f2 ", "map", "val/AP/raccoon"]}"#),
    )
    .await;
    let v = json(b);
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["success"], true);
    assert_eq!(
        v["columns"],
        serde_json::json!(["f2", "map", "val/AP/raccoon"])
    );
    assert_eq!(
        state.config_read().training.metric_columns,
        ["f2", "map", "val/AP/raccoon"]
    );
    let (_, text) = get_text(&state, "/v1/training").await;
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(
        v["metric_columns"],
        serde_json::json!(["f2", "map", "val/AP/raccoon"])
    );

    // Bad lists: 400, nothing changes.
    let too_many: Vec<String> = (0..21).map(|i| format!("m{i}")).collect();
    for bad in [
        r#"{"columns": []}"#.to_string(),
        r#"{"columns": ["map", ""]}"#.to_string(),
        r#"{"columns": ["map", "map"]}"#.to_string(),
        format!(r#"{{"columns": ["{}"]}}"#, "x".repeat(65)),
        serde_json::json!({ "columns": too_many }).to_string(),
        r#"{"columns": "map"}"#.to_string(),
    ] {
        let (s, _, b) = call(&state, post(&bad)).await;
        let v = json(b);
        assert_eq!(s, StatusCode::BAD_REQUEST, "{bad}: {v}");
        assert_eq!(v["success"], false);
        assert!(v["message"].as_str().is_some_and(|m| !m.is_empty()));
    }
    assert_eq!(
        state.config_read().training.metric_columns,
        ["f2", "map", "val/AP/raccoon"]
    );
    // Not JSON: 415.
    let form = Request::post("/v1/training/metric-columns")
        .header(header::CONTENT_TYPE, "application/x-www-form-urlencoded")
        .body(Body::from("columns=map"))
        .unwrap();
    let (s, _, _) = call(&state, form).await;
    assert_eq!(s, StatusCode::UNSUPPORTED_MEDIA_TYPE);
    // Admin only: a request that is not signed in is refused.
    let (s, _, _) = call(
        &state,
        crate::server::test_util::anonymous(from_peer(post(r#"{"columns": ["map"]}"#), REMOTE)),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    // The default list is stored as empty (not written to the config file).
    let (s, _, _) = call(
        &state,
        post(r#"{"columns": ["map", "map50", "iou", "f1", "f2", "precision", "recall", "loss"]}"#),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(state.config_read().training.metric_columns.is_empty());
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn busy_jobs_show_pause_resume_and_cancel() {
    use crate::server::test_util::{call, get_text, post_form, test_state_with};
    use crate::training::task::{TaskKind, TaskRecord};
    let root = std::env::temp_dir().join(format!("bop-train-pause-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let _queue = own_queue(&state).await;
    let svc = service(&Shared::from(state.clone()));
    let rec = |id: &str, st: RunState| TaskRecord {
        id: id.into(),
        kind: TaskKind::Autolabel,
        title: format!("auto-label {id}"),
        created: String::new(),
        state: st,
        stage: Some("labelling".into()),
        step: 2,
        steps: 3,
        percent: 40,
        detail: "5 of 9 frames".into(),
        log: vec![],
        summary: None,
        result: None,
        error: None,
        finished: None,
        after: None,
        uses: Vec::new(),
    };
    let (run, paused) = ("pause-test-running", "pause-test-paused");
    svc.put_task_record(rec(run, RunState::Running));
    svc.put_task_record(rec(paused, RunState::Paused));
    let (s, page) = get_text(&state, "/train/train").await;
    assert_eq!(s, StatusCode::OK);
    let row = |id: &str| {
        let i = page
            .find(&format!("data-task=\"{id}\""))
            .expect("the job's row");
        let j = i + page[i..].find("</tr>").unwrap();
        page[i..j].to_string()
    };
    let (r, p) = (row(run), row(paused));
    let shown =
        |form: &str| format!("action=\"/training/{form}\" class=\"inline\" data-field=\"{form}\">");
    let hidden = |form: &str| {
        format!("action=\"/training/{form}\" class=\"inline\" data-field=\"{form}\" hidden>")
    };
    assert!(
        r.contains(&shown("pause")) && r.contains(&hidden("resume")),
        "{r}"
    );
    assert!(
        p.contains(&hidden("pause")) && p.contains(&shown("resume")),
        "{p}"
    );
    assert!(r.contains("/training/cancel") && p.contains("/training/cancel"));
    assert!(p.contains("paused (labelling 2/3): 5 of 9 frames"), "{p}");
    assert!(r.contains("GPU memory stays in use"), "the tooltip");
    let (_, json) = get_text(&state, "/v1/training").await;
    let v: serde_json::Value = serde_json::from_str(&json).unwrap();
    let st = v["tasks"]
        .as_array()
        .unwrap()
        .iter()
        .find(|t| t["id"] == paused)
        .map(|t| t["state"].clone());
    assert_eq!(st, Some(serde_json::json!("paused")));
    // Not the job on the queue's thread: nothing to pause or resume.
    let (s, _, b) = call(
        &state,
        post_form("/training/pause", &format!("id={run}"), true),
    )
    .await;
    assert_eq!(s, StatusCode::OK);
    assert!(String::from_utf8(b).unwrap().contains("No running job"));
    let (_, _, b) = call(
        &state,
        post_form("/training/resume", &format!("id={paused}"), true),
    )
    .await;
    assert!(String::from_utf8(b).unwrap().contains("No paused job"));
    // Finish now: only a training run on the queue's thread.
    for id in [run, "no-such-run"] {
        let (s, _, b) = call(
            &state,
            post_form("/training/finish", &format!("id={id}"), true),
        )
        .await;
        assert_eq!(s, StatusCode::OK);
        let b = String::from_utf8(b).unwrap();
        assert!(b.contains("Not finished:"), "{id}");
    }
    for id in [run, paused] {
        svc.put_task_record(rec(id, RunState::Cancelled));
        svc.remove(id).unwrap();
    }
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn scenes_times_models_queue_a_comparison_group() {
    let f = form(&[
        ("model", "yolo26n"),
        ("model", "rfdetr-nano"),
        ("scene", "day"),
        ("scene", "night+ir"),
        ("scene", "all"),
        ("scene", "ir+night"),
        ("scene", ""),
    ]);
    let (reqs, name) = requests_from_form(&f, &mac(), &Default::default()).unwrap();
    assert_eq!(name, None);
    let got: Vec<(String, &str)> = reqs
        .iter()
        .map(|r| (r.recipe.label(), r.scene.as_str()))
        .collect();
    assert_eq!(
        got,
        [
            ("yolo26n".to_string(), "day"),
            ("yolo26n".to_string(), "night+ir"),
            ("yolo26n".to_string(), "all"),
            ("rfdetr-nano".to_string(), "day"),
            ("rfdetr-nano".to_string(), "night+ir"),
            ("rfdetr-nano".to_string(), "all"),
        ]
    );
    let group = reqs[0].compare_group.clone().unwrap();
    assert!(group.starts_with("cmp-") && group.len() == "cmp-20261027-120000".len());
    assert!(
        reqs.iter()
            .all(|r| r.compare_group.as_ref() == Some(&group))
    );
    // Every run gets its own default name within the same stamp.
    let mut names: Vec<String> = reqs
        .iter()
        .map(|r| r.recipe.default_name(r.scene.label(), "20261027-1200"))
        .collect();
    assert_eq!(
        names[..3],
        [
            "day-yolo26n-20261027-1200",
            "night-yolo26n-20261027-1200",
            "custom-yolo26n-20261027-1200"
        ]
    );
    names.sort();
    names.dedup();
    assert_eq!(names.len(), 6);
    // One scene: no group, even with several models; none ticked = all.
    let (reqs, _) = requests_from_form(
        &form(&[
            ("model", "yolo26n"),
            ("model", "yolo26s"),
            ("scene", "night"),
        ]),
        &mac(),
        &Default::default(),
    )
    .unwrap();
    assert!(reqs.iter().all(|r| r.compare_group.is_none()));
    assert_eq!(reqs[0].scene, SceneFilter::new(false, true, false));
    let (reqs, _) =
        requests_from_form(&form(&[("model", "yolo26n")]), &mac(), &Default::default()).unwrap();
    assert_eq!(reqs[0].scene, SceneFilter::ALL_SCENES);
    assert_eq!(reqs[0].compare_group, None);
    // A typed name: one run only, scenes included.
    let e = requests_from_form(
        &form(&[
            ("model", "yolo26n"),
            ("scene", "day"),
            ("scene", "all"),
            ("name", "x"),
        ]),
        &mac(),
        &Default::default(),
    )
    .unwrap_err();
    assert!(e.to_string().contains("one run only"), "{e}");
    let (reqs, name) = requests_from_form(
        &form(&[("model", "yolo26n"), ("scene", "day"), ("name", "x")]),
        &mac(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!((reqs.len(), name.as_deref()), (1, Some("x")));
    let e = requests_from_form(
        &form(&[("model", "yolo26n"), ("scene", "dusk")]),
        &mac(),
        &Default::default(),
    )
    .unwrap_err();
    assert!(e.to_string().starts_with("scene: 'dusk'"), "{e}");
}

#[test]
fn classes_repeat_or_comma_and_excluded_parses() {
    let (reqs, _) = requests_from_form(
        &form(&[
            ("model", "yolo26n"),
            ("classes", "person"),
            ("classes", "car, dog"),
            ("classes", " person "),
            ("classes", ""),
            ("excluded", "drop-frame"),
        ]),
        &mac(),
        &Default::default(),
    )
    .unwrap();
    assert_eq!(
        reqs[0].classes.as_deref(),
        Some(&["person".to_string(), "car".into(), "dog".into()][..])
    );
    assert_eq!(reqs[0].excluded, Excluded::DropFrame);
    let (reqs, _) =
        requests_from_form(&form(&[("model", "yolo26n")]), &mac(), &Default::default()).unwrap();
    assert_eq!(reqs[0].classes, None);
    assert_eq!(reqs[0].excluded, Excluded::Background);
    let e = requests_from_form(
        &form(&[("model", "yolo26n"), ("excluded", "maybe")]),
        &mac(),
        &Default::default(),
    )
    .unwrap_err();
    assert!(
        e.to_string()
            .starts_with("objects of unticked classes: 'maybe'"),
        "{e}"
    );
}

#[test]
fn rename_repeats_or_commas_and_empty_to_drops() {
    let (reqs, _) = requests_from_form(
        &form(&[
            ("model", "yolo26n"),
            ("model", "yolo26s"),
            ("scene", "day"),
            ("scene", "night"),
            ("seeds", "2"),
            ("rename", "truck=car"),
            ("rename", "van = car, tree="),
            ("rename", "truck=bus"),
            ("rename", ""),
        ]),
        &mac(),
        &Default::default(),
    )
    .unwrap();
    // Every model x scene x seed request carries the merges; a later duplicate wins.
    assert_eq!(reqs.len(), 8);
    for r in &reqs {
        assert_eq!(
            r.rename,
            BTreeMap::from([
                ("truck".to_string(), "bus".to_string()),
                ("van".to_string(), "car".to_string()),
                ("tree".to_string(), String::new()),
            ])
        );
    }
    let (reqs, _) =
        requests_from_form(&form(&[("model", "yolo26n")]), &mac(), &Default::default()).unwrap();
    assert!(reqs[0].rename.is_empty());
    let e = requests_from_form(
        &form(&[("model", "yolo26n"), ("rename", "truck=car, =x")]),
        &mac(),
        &Default::default(),
    )
    .unwrap_err();
    assert_eq!(e.to_string(), "class merges: '=x' is not from=to");
    let e = rename_of(&form(&[("rename", "truck")])).unwrap_err();
    assert!(e.to_string().starts_with("class merges: 'truck'"), "{e}");
}

#[test]
fn mixed_renames_warn_in_a_comparison_and_equal_ones_do_not() {
    let mk = |rename: &[(&str, &str)]| {
        let mut r = crate::training::compare::CompareRun {
            name: "n".into(),
            model: "yolo26n".into(),
            scene: SceneFilter::ALL_SCENES,
            classes: vec!["car".into()],
            rename: rename
                .iter()
                .map(|(a, b)| (a.to_string(), b.to_string()))
                .collect(),
            excluded: Excluded::Background,
            ignore: None,
            negative_share: None,
            static_cap: None,
            workers: None,
            balance: Default::default(),
            best_epoch: None,
            stopped_epoch: None,
            seed: None,
            select_metric: Default::default(),
            select_classes: None,
            bench: None,
        };
        r.name = format!("r{}", rename.len());
        r
    };
    let merged = [("truck", "car")];
    let c = compare::compare(&[mk(&merged), mk(&merged), mk(&[])]);
    assert!(
        c.warnings
            .iter()
            .any(|w| w.contains("merge classes differently")
                && w.contains("truck \u{2192} car on 2 of 3")),
        "{:?}",
        c.warnings
    );
    for runs in [[mk(&merged), mk(&merged)], [mk(&[]), mk(&[])]] {
        let c = compare::compare(&runs);
        assert!(
            !c.warnings.iter().any(|w| w.contains("merge classes")),
            "{:?}",
            c.warnings
        );
    }
    assert_eq!(merges_text(&BTreeMap::new()), "");
    assert_eq!(
        merges_text(&BTreeMap::from([
            ("truck".to_string(), "car".to_string()),
            ("tree".to_string(), String::new()),
        ])),
        "merged: truck \u{2192} car; dropped: tree"
    );
}

fn truck_export(dir: &std::path::Path) {
    std::fs::create_dir_all(dir.join("images")).unwrap();
    std::fs::create_dir_all(dir.join("labels")).unwrap();
    std::fs::write(dir.join("classes.txt"), "person\ncar\ntruck\n").unwrap();
    for cam in ["a", "b", "c"] {
        for i in 0..12 {
            let stem = format!("day.{cam}.2026-10-01.f{i}");
            std::fs::write(dir.join("images").join(format!("{stem}.jpg")), b"x").unwrap();
            let label = match i % 3 {
                0 => "1 0.5 0.5 0.2 0.4\n",
                1 => "2 0.5 0.5 0.2 0.4\n",
                _ => "0 0.5 0.5 0.2 0.4\n",
            };
            std::fs::write(dir.join("labels").join(format!("{stem}.txt")), label).unwrap();
        }
    }
}

#[tokio::test]
async fn merged_classes_count_in_the_preview_and_the_class_list() {
    use crate::server::test_util::{call, get_text, post_form, test_state_with};
    let root = std::env::temp_dir().join(format!("bop-train-mrg-{}", uuid::Uuid::new_v4()));
    let export = root.join("exports/review-1");
    truck_export(&export);
    let (r2, e2) = (root.clone(), export.clone());
    let state = test_state_with(move |c| {
        c.download_dir = Some(r2);
        c.training.datasets = vec![e2];
    });
    let enc = super::super::encode_uri_component(&export.display().to_string());
    let total = |v: &serde_json::Value, name: &str| -> Option<u64> {
        v["runs"][0]["preflight"]["classes"]
            .as_array()
            .unwrap()
            .iter()
            .find(|c| c["name"] == name)
            .map(|c| {
                ["train", "valid", "heldout"]
                    .iter()
                    .map(|k| c[*k].as_u64().unwrap())
                    .sum()
            })
    };
    let preview = |extra: &str| {
        let state = &state;
        let body = format!("dataset={enc}&scene=all{extra}");
        async move {
            let (s, _, b) = call(
                state,
                post_form("/v1/training/balance-preview", &body, false),
            )
            .await;
            assert_eq!(s, StatusCode::OK, "{}", String::from_utf8_lossy(&b));
            serde_json::from_slice::<serde_json::Value>(&b).unwrap()
        }
    };
    let plain = preview("").await;
    let (car0, truck0) = (
        total(&plain, "car").unwrap(),
        total(&plain, "truck").unwrap(),
    );
    assert!(car0 > 0 && truck0 > 0, "{plain}");
    let merged = preview("&rename=truck%3Dcar").await;
    assert_eq!(total(&merged, "car"), Some(car0 + truck0), "{merged}");
    assert_eq!(total(&merged, "truck"), None, "{merged}");

    // The class picker's answer: car summed, truck out of the order, kept as a source.
    let (s, text) = get_text(
        &state,
        &format!("/v1/training/dataset-classes?dataset={enc}&rename=truck%3Dcar"),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    let car = v["classes"]
        .as_array()
        .unwrap()
        .iter()
        .find(|c| c["name"] == "car")
        .unwrap();
    assert_eq!(car["boxes"], 24);
    assert!(
        v["classes"]
            .as_array()
            .unwrap()
            .iter()
            .all(|c| c["name"] != "truck")
    );
    assert_eq!(v["order"], serde_json::json!(["person", "car"]));
    assert_eq!(
        v["source_classes"],
        serde_json::json!([
            {"name": "person", "boxes": 12},
            {"name": "car", "boxes": 12},
            {"name": "truck", "boxes": 12},
        ])
    );
    // A dropped class leaves the order too; a bad merge is a 400 naming it.
    let (_, text) = get_text(
        &state,
        &format!("/v1/training/dataset-classes?dataset={enc}&rename=truck%3D"),
    )
    .await;
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["order"], serde_json::json!(["person", "car"]));
    let (s, text) = get_text(
        &state,
        &format!("/v1/training/dataset-classes?dataset={enc}&rename=%3Dcar"),
    )
    .await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{text}");
    assert!(
        text.contains("class merges: '=car' is not from=to"),
        "{text}"
    );
}

#[test]
fn default_names_get_a_suffix_when_taken() {
    let taken = ["day-yolo26n-1", "day-yolo26n-1-2"];
    let t = |n: &str| taken.contains(&n);
    assert_eq!(unique_name("night-yolo26n-1", t), "night-yolo26n-1");
    assert_eq!(unique_name("day-yolo26n-1", t), "day-yolo26n-1-3");
    assert!(recipe::valid_model_name(&unique_name("day-yolo26n-1", t)).is_ok());
}

/// A tiny export: `person` boxes in day and night frames, `car` in one IR frame.
fn tiny_export(dir: &std::path::Path) {
    std::fs::create_dir_all(dir.join("images")).unwrap();
    std::fs::create_dir_all(dir.join("labels")).unwrap();
    std::fs::write(dir.join("classes.txt"), "person\ncar\nbus\n").unwrap();
    for (stem, labels) in [
        (
            "day.cam1.2026-10-01.a",
            "0 0.5 0.5 0.2 0.4\n0 0.2 0.2 0.1 0.2\n",
        ),
        ("night.cam1.2026-10-02.b", "0 0.5 0.5 0.2 0.4\n"),
        ("ir.cam2.2026-10-02.c", "1 0.5 0.5 0.3 0.2\n"),
        ("plain", ""),
    ] {
        std::fs::write(dir.join("images").join(format!("{stem}.jpg")), b"x").unwrap();
        std::fs::write(dir.join("labels").join(format!("{stem}.txt")), labels).unwrap();
    }
}

#[tokio::test]
async fn dataset_classes_lists_counts_per_scene_and_refuses_other_folders() {
    use crate::server::test_util::{REMOTE, call, from_peer, get_text, test_state_with};
    let root = std::env::temp_dir().join(format!("bop-train-cls-{}", uuid::Uuid::new_v4()));
    let export = root.join("exports/review-1");
    tiny_export(&export);
    let (r2, e2) = (root.clone(), export.clone());
    let state = test_state_with(move |c| {
        c.download_dir = Some(r2);
        c.training.datasets = vec![e2];
    });
    let enc = super::super::encode_uri_component(&export.display().to_string());
    let (s, text) = get_text(
        &state,
        &format!("/v1/training/dataset-classes?dataset={enc}&dataset=%2Fno%2Fsuch%2Fexport"),
    )
    .await;
    assert_eq!(s, StatusCode::OK, "{text}");
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(v["success"], true);
    let classes = v["classes"].as_array().unwrap();
    assert_eq!(classes[0]["name"], "person");
    assert_eq!(classes[0]["boxes"], 3);
    assert_eq!(classes[0]["frames"], 2);
    assert_eq!(
        classes[0]["by_scene"],
        serde_json::json!({"day": 2, "night": 1, "ir": 0, "unknown": 0})
    );
    assert_eq!(classes[1]["name"], "car");
    assert_eq!(classes[1]["by_scene"]["ir"], 1);
    // A listed class without a box is still offered; the order is the class list's.
    assert_eq!(classes[2]["name"], "bus");
    assert_eq!(classes[2]["boxes"], 0);
    assert_eq!(v["order"], serde_json::json!(["person", "car", "bus"]));
    assert_eq!(
        v["frames"],
        serde_json::json!({"day": 1, "night": 1, "ir": 1, "unknown": 1})
    );
    // The folder outside the offered and browsable ones is refused and named.
    let errors = v["errors"].as_array().unwrap();
    assert_eq!(errors.len(), 1, "{v}");
    assert_eq!(errors[0]["dataset"], "/no/such/export");
    assert!(errors[0]["message"].as_str().is_some_and(|m| !m.is_empty()));
    // Nothing ticked: nothing listed.
    let (_, text) = get_text(&state, "/v1/training/dataset-classes").await;
    let v: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert!(v["classes"].as_array().unwrap().is_empty());
    // Admin only.
    let req = axum::http::Request::get("/v1/training/dataset-classes")
        .body(axum::body::Body::empty())
        .unwrap();
    let (s, _, _) = call(
        &state,
        crate::server::test_util::anonymous(from_peer(req, REMOTE)),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let _ = std::fs::remove_dir_all(&root);
}

fn scene_ap(v: f64) -> crate::training::job::SceneAp {
    crate::training::job::SceneAp {
        ap50: Some(v + 0.2),
        ap50_95: Some(v),
        images: 10,
        ap_basis: None,
    }
}

/// A run record of comparison `group` (benchmarked with day / night / IR AP when `ap`).
pub(in crate::server) fn record(
    id: &str,
    scene: SceneFilter,
    group: Option<&str>,
    st: RunState,
    ap: Option<(f64, f64, f64)>,
) -> RunRecord {
    use crate::resources::catalog::TrainFamily;
    let request = TrainRequest {
        recipe: TrainRecipe::new(TrainFamily::Yolo26, "n").unwrap(),
        datasets: vec!["/exports/a".into()],
        include_public: false,
        scene,
        classes: Some(vec!["person".into()]),
        rename: Default::default(),
        benchmark_sets: Vec::new(),
        benchmark_devices: Vec::new(),
        allow_large: false,
        install_dir: None,
        gpu: None,
        excluded: Excluded::Background,
        balance: Default::default(),
        compare_group: group.map(str::to_string),
        ignore_memory_check: false,
        train_seed: None,
    };
    let benchmark = ap.map(|(d, n, i)| BenchSummary {
        trained_on_share: None,
        datasets: vec![format!("heldout-{id}")],
        device: Some("ort:coreml".into()),
        ms_p50: Some(10.0 + d),
        ap50: Some(d + 0.2),
        ap50_95: Some(d),
        images: 30,
        by_scene: [
            ("day", d),
            ("night", n),
            ("ir", i),
            ("all", (d + n + i) / 3.0),
        ]
        .into_iter()
        .map(|(k, v)| (k.to_string(), scene_ap(v)))
        .collect(),
        in_scope: Some(scene_ap(d.max(n))),
        heldout_fingerprint: Some("f1".into()),
        threshold: Some(0.5),
        person_recall: Some(d),
        vehicle_recall: None,
        empty_frames: Some(40),
        empty_frame_alerts: Some((10.0 * n).round() as usize),
        random: None,
        ap_basis: None,
        person_free_frames: None,
        person_false_alerts: None,
        recall_cells: Default::default(),
    });
    RunRecord {
        baseline: None,
        id: id.into(),
        name: format!("{}-yolo26n-{id}", scene.label()),
        created: String::new(),
        request,
        device: "mps".into(),
        state: st,
        stage: None,
        step: 0,
        steps: 8,
        percent: 0,
        detail: String::new(),
        progress: Some(crate::training::progress::TrainProgress {
            epoch: 27,
            epochs: 100,
            patience: 15,
            stopped: Some(crate::training::progress::Stop {
                reason: "converged".into(),
                epoch: 27,
                best_epoch: 12,
            }),
            ..Default::default()
        }),
        log: Vec::new(),
        dataset: None,
        output: None,
        registered: None,
        benchmark,
        benchmark_error: None,
        error: None,
        finished: None,
        run_dir: PathBuf::from("/runs").join(id),
        retrying: false,
        resources: None,
    }
}

#[test]
fn the_switch_preview_compares_the_default_with_the_trained_model() {
    use crate::server::test_util::test_state_with;
    use crate::training::job::Baseline;
    let root = std::env::temp_dir().join(format!("bop-train-prev-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(root.join("models")).unwrap();
    let yaml = |name: &str, classes: &str| {
        let p = root.join("models").join(name);
        std::fs::write(&p, format!("NAMES:\n{classes}")).unwrap();
        p
    };
    let old = yaml("old.yaml", "  - person\n  - dog\n");
    let new = yaml("trained.yaml", "  - person\n");
    let onnx = root.join("models/trained.onnx");
    std::fs::write(&onnx, vec![1u8; 2048]).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| {
        c.download_dir = Some(r2);
        c.models[0].classes = Some(old);
        c.models[0].confidence_threshold = Some(0.7);
    });
    let mut rec = record("r1", SceneFilter::DAY, None, RunState::Done, None);
    rec.output = Some(TrainOutput {
        name: "trained".into(),
        onnx,
        yaml: new,
        provenance: root.join("models/trained.provenance.json"),
        family: ModelFamilyKind::Yolo26,
        classes: vec!["person".into()],
        license: "AGPL-3.0".into(),
        heldout: None,
        device: "mps".into(),
        train_seconds: 1.0,
    });
    rec.benchmark = Some(BenchSummary {
        datasets: vec!["heldout-trained".into()],
        images: 20,
        trained_on_share: Some(0.5),
        ..BenchSummary::default()
    });
    rec.baseline = Some(Baseline {
        model: "ipcam-general".into(),
        summary: None,
        note: Some("ipcam-general is not loaded".into()),
        ..Baseline::default()
    });
    // The model is not in the config yet: the preview does not add it.
    let p = super::super::training_switch::preview_of(&state, &rec).unwrap();
    assert!(
        !state
            .config_read()
            .models
            .iter()
            .any(|m| m.effective_name() == "trained")
    );
    let v = serde_json::to_value(&p).unwrap();
    assert_eq!(v["already_default"], false);
    assert_eq!(
        (v["current"].as_str(), v["new"].as_str()),
        (Some("ipcam-general"), Some("trained"))
    );
    assert_eq!(v["classes"]["lost"], serde_json::json!(["dog"]));
    assert!(
        v["rows"][0]["current"]
            .as_str()
            .unwrap()
            .starts_with("0.70"),
        "{v}"
    );
    assert!(v["copy"]["copied"][0]["value"] == "0.70", "{v}");
    assert!(
        v["no_suggestion"]
            .as_str()
            .unwrap()
            .contains("no benchmark yet"),
        "{v}"
    );
    assert!(v["compare"].as_str().unwrap().contains("not loaded"), "{v}");
    assert!(v["trained_on"].as_str().unwrap().contains("50%"), "{v}");
    let _ = std::fs::remove_dir_all(&root);
}

fn group_records() -> Vec<RunRecord> {
    let mut running = record("r9", SceneFilter::DAY, None, RunState::Running, None);
    running.stage = Some(crate::training::job::TrainStage::Train);
    running.progress = Some(crate::training::progress::TrainProgress {
        epoch: 4,
        epochs: 100,
        patience: 15,
        since_best: 1,
        extension: Some(crate::training::progress::ExtensionNotice {
            text: "Early stop reached at epoch 3 (best epoch 2); validation still improving."
                .into(),
            keep_epoch: 2,
            extension: 1,
        }),
        ..Default::default()
    });
    let mut queued = record("r8", SceneFilter::DAY, None, RunState::Queued, None);
    queued.progress = None;
    vec![
        record(
            "old1",
            SceneFilter::DAY,
            Some("cmp-old"),
            RunState::Done,
            Some((0.5, 0.1, 0.1)),
        ),
        record(
            "old2",
            SceneFilter::ALL_SCENES,
            Some("cmp-old"),
            RunState::Done,
            Some((0.4, 0.3, 0.2)),
        ),
        record(
            "a",
            SceneFilter::DAY,
            Some("cmp-new"),
            RunState::Done,
            Some((0.6, 0.2, 0.1)),
        ),
        record(
            "b",
            SceneFilter::NIGHT_IR,
            Some("cmp-new"),
            RunState::Done,
            Some((0.3, 0.5, 0.45)),
        ),
        record(
            "c",
            SceneFilter::ALL_SCENES,
            Some("cmp-new"),
            RunState::Running,
            None,
        ),
        record(
            "solo",
            SceneFilter::DAY,
            Some("cmp-solo"),
            RunState::Done,
            Some((0.1, 0.1, 0.1)),
        ),
        queued,
        running,
    ]
}

#[test]
fn comparison_groups_rows_and_json() {
    let recs = group_records();
    let groups = compare_groups(&recs);
    let names: Vec<&str> = groups.iter().map(|(g, _)| g.as_str()).collect();
    assert_eq!(
        names,
        ["cmp-new", "cmp-old"],
        "newest first, single runs left out"
    );
    let (runs, c) = group_comparison(&recs, "cmp-new").unwrap();
    assert_eq!(runs.len(), 3);
    assert!(c.comparable);
    assert_eq!(c.rows.len(), 4, "3 runs and the routed row");
    assert!(c.rows[3].routed);
    assert_eq!(c.best_per_column["day"], 0);
    assert_eq!(c.best_per_column["night"], 1);
    assert_eq!(c.rows[0].best_epoch, Some(12));
    assert_eq!(c.rows[0].stopped_epoch, Some(27));
    assert_eq!(c.rows[2].day.ap50_95, None, "the running run has no AP yet");
    assert!(group_comparison(&recs, "cmp-none").is_none());
    let cards = compare_cards(&recs);
    assert_eq!(cards.len(), 2);
    assert!(cards[0].open && !cards[1].open);
    let r = &cards[0].rows;
    assert!(r[0].cells[0].best && !r[1].cells[0].best);
    assert_eq!(r[0].cells[0].ap, "0.600");
    assert_eq!(r[0].cells[0].ap50, "0.800");
    assert_eq!(r[0].epochs, "best 12 / stopped 27");
    assert_eq!(r[2].cells[0].ap, "\u{2014}");
    assert_eq!(r[2].state, "running");
    assert!(r[3].routed && !r[3].cells[1].best);
    // Alerting columns: person recall, vehicle recall (none recorded), false alerts, false
    // person alerts (none recorded).
    let texts = |k: usize| -> Vec<&str> { r[k].alerts.iter().map(|a| a.text.as_str()).collect() };
    assert_eq!(texts(0), ["0.60", "\u{2014}", "2/40", "\u{2014}"]);
    assert_eq!(texts(1), ["0.30", "\u{2014}", "5/40", "\u{2014}"]);
    assert!(r[0].alerts[0].best && r[0].alerts[2].best && !r[0].alerts[1].best);
    assert!(!r[1].alerts[0].best && !r[1].alerts[2].best);
    assert_eq!(r[0].alerts[0].title, "recall at threshold 0.50, IoU 0.5");
    assert_eq!(
        r[0].alerts[2].title,
        "2 of 40 frames without any object alerted (5.0%) at threshold 0.50"
    );
    assert_eq!(texts(2), ["\u{2014}"; 4], "the running run");
    assert_eq!(texts(3), ["\u{2014}"; 4], "the routed row");
    // False person alerts, when recorded: the count and what it covers.
    let mut with_person = recs.clone();
    // Records `a` and `b` of the newest group.
    for (k, (a, f)) in [(2, (2, 210)), (3, (5, 200))] {
        let b = with_person[k].benchmark.as_mut().unwrap();
        b.person_false_alerts = Some(a);
        b.person_free_frames = Some(f);
    }
    let cards = compare_cards(&with_person);
    let pr = &cards[0].rows;
    assert_eq!(pr[0].alerts[3].text, "2/210");
    assert_eq!(
        pr[0].alerts[3].title,
        "2 of 210 frames without a person had a person detection (1.0%) at threshold 0.50"
    );
    assert!(pr[0].alerts[3].best && !pr[1].alerts[3].best);
    assert!(r[3].alerts.iter().all(|a| !a.best && a.title.is_empty()));
    // A different held-out set: not comparable, said so.
    let mut recs = recs;
    recs[3].benchmark.as_mut().unwrap().heldout_fingerprint = Some("f2".into());
    let cards = compare_cards(&recs);
    assert!(!cards[0].comparable);
    assert!(!cards[0].warnings.is_empty());
    // The JSON shape.
    let (_, c) = group_comparison(&recs, "cmp-old").unwrap();
    let v = serde_json::to_value(&c).unwrap();
    assert_eq!(v["comparable"], true);
    assert_eq!(v["rows"][0]["day"]["ap50_95"], 0.5);
    assert_eq!(v["best_per_column"]["night"], 1);
    assert_eq!(v["random_column"], false);
    // No random picks: no column.
    assert!(!cards[0].random && cards[0].random_tip.is_empty());
}

/// `group_records` with `n` held-out random picks on the benchmarked runs of `cmp-new`.
fn records_with_random_picks(n: usize) -> Vec<RunRecord> {
    let mut recs = group_records();
    for (id, ap) in [("a", 0.31), ("b", 0.42)] {
        let r = recs.iter_mut().find(|r| r.id == id).unwrap();
        r.benchmark.as_mut().unwrap().random = Some(crate::training::job::SubsetScore {
            images: n,
            gt: 85,
            ap50: Some(ap + 0.2),
            ap50_95: Some(ap),
            recall: Some(0.7),
            ap_basis: None,
        });
    }
    recs
}

#[test]
fn the_random_picks_column_follows_the_fifty_frame_rule() {
    let cards = compare_cards(&records_with_random_picks(60));
    let card = &cards[0];
    assert!(card.random);
    assert_eq!(
        card.random_tip,
        "AP50-95 on held-out frames the queue picked at random (60 frames, 85 objects): \
         nobody chose them, so this is the least biased score. Hidden under 50 frames."
    );
    let (a, b) = (&card.rows[0].random, &card.rows[1].random);
    assert_eq!((a.text.as_str(), b.text.as_str()), ("0.310", "0.420"));
    assert!(!a.best && b.best);
    assert_eq!(
        a.title,
        "AP50-95 on 60 held-out frames picked at random (85 objects), AP50 0.510, recall 0.70 \
         at threshold 0.50"
    );
    // The running run and the routed row: dashes, never the best.
    for r in &card.rows[2..] {
        assert_eq!(r.random.text, "\u{2014}");
        assert!(r.random.title.is_empty() && !r.random.best);
    }
    // Under 50 in a run: hidden.
    let cards = compare_cards(&records_with_random_picks(49));
    assert!(!cards[0].random && cards[0].random_tip.is_empty());
    assert!(!cards[0].rows[1].random.best);
}

#[tokio::test]
async fn the_comparison_card_draws_the_random_picks_column_only_when_shown() {
    use crate::server::test_util::test_state_with;
    use askama::Template;
    let root = std::env::temp_dir().join(format!("bop-train-rnd-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let render = |recs: Vec<RunRecord>| {
        let page = view_with(
            &state,
            MODELS,
            None,
            None,
            recs,
            crate::training::capability::host_for_tests(),
        )
        .render()
        .unwrap();
        let i = page.find("id=\"compare\"").expect("the Comparison card");
        page[i..].to_string()
    };
    let shown = render(records_with_random_picks(60));
    assert!(
        shown.contains(
            "<th class=\"num\"><button type=\"button\" class=\"term\" data-tip=\"AP50-95 on \
             held-out frames the queue picked at random (60 frames, 85 objects): nobody chose \
             them, so this is the least biased score. Hidden under 50 frames.\">Random \
             picks</button></th>"
        ),
        "{shown}"
    );
    assert!(
        shown.contains(
            "<td class=\"num best-val\"><button type=\"button\" class=\"term\" \
             data-tip=\"AP50-95 on 60 held-out frames picked at random (85 objects), AP50 \
             0.620, recall 0.70 at threshold 0.50\">0.420</button></td>"
        ),
        "{shown}"
    );
    // The header and the run rows keep the same number of cells.
    let section = shown.split("cmp-old").next().unwrap();
    let heads = section.matches("<th>").count() + section.matches("<th class").count();
    let row = section.split("<tbody>").nth(1).unwrap();
    let first = row.split("</tr>").next().unwrap();
    assert_eq!(first.matches("<td").count(), heads, "{first}");
    let hidden = render(records_with_random_picks(49));
    assert!(!hidden.contains(">Random picks</button>"), "{hidden}");
    assert!(!hidden.contains("picked at random ("), "{hidden}");
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn a_failed_run_shows_its_warnings_files_and_retry_export() {
    use crate::server::test_util::{call, post_form, test_state_with};
    use askama::Template;
    // The retry route moves the process's queue to this test's folder.
    let _queue = QUEUE_TEST_LOCK.lock().await;
    let root = std::env::temp_dir().join(format!("bop-train-retry-{}", uuid::Uuid::new_v4()));
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let mut recs = group_records();
    let dir = root.join("training").join("r9");
    let r9 = recs.iter_mut().find(|r| r.id == "r9").unwrap();
    r9.state = RunState::Failed;
    r9.run_dir.clone_from(&dir);
    r9.progress = None;
    r9.dataset = Some(crate::training::service::DatasetCounts {
        warnings: vec!["no frame is held out: <check>".into()],
        ..Default::default()
    });
    std::fs::create_dir_all(dir.join("work/run/weights")).unwrap();
    std::fs::create_dir_all(dir.join("data")).unwrap();
    std::fs::write(dir.join("data/dataset.json"), "{}").unwrap();
    std::fs::write(dir.join("work/run/weights/best.pt"), vec![0u8; 3000]).unwrap();
    let cap = crate::training::capability::host_for_tests();
    let page = view_with(&state, TRAIN, None, None, recs.clone(), cap.clone())
        .render()
        .unwrap();
    let row = page.split("<tr data-run=\"r9\"").nth(1).unwrap();
    let row = row.split("tr-metrics-row").next().unwrap();
    assert!(
        row.contains("no frame is held out: &#60;check&#62;")
            || row.contains("no frame is held out: &lt;check&gt;"),
        "{row}"
    );
    assert!(!row.contains("data-field=\"warnings\" hidden"), "{row}");
    assert!(row.contains("action=\"/training/retry-export\""), "{row}");
    assert!(
        row.contains("Run files: 3.0 kB on disk, 3.0 kB of it checkpoints")
            || row.contains("Run files: 3 kB on disk"),
        "{row}"
    );
    assert!(
        row.contains("A successful retry or Remove frees them"),
        "{row}"
    );
    assert!(row.contains("Remove and delete files"), "{row}");
    assert!(row.contains("data-confirm="), "{row}");
    assert!(row.contains("name=\"delete_files\""), "{row}");
    // The checkpoint gone: no button, the files line stays.
    std::fs::remove_dir_all(dir.join("work")).unwrap();
    let page = view_with(&state, TRAIN, None, None, recs, cap)
        .render()
        .unwrap();
    let row = page.split("<tr data-run=\"r9\"").nth(1).unwrap();
    let row = row.split("tr-metrics-row").next().unwrap();
    assert!(!row.contains("/training/retry-export"), "{row}");
    // The route names what is wrong.
    let (_, _, b) = call(&state, post_form("/training/retry-export", "id=nope", true)).await;
    assert!(String::from_utf8(b).unwrap().contains("no run nope"));
    let _ = std::fs::remove_dir_all(&root);
}

/// The Models tab's results list: groups of runs whose held-out scores compare (fingerprint,
/// classes, merges), newest group first, each led by its header row and best first (person
/// R, vehicle R, AP50-95), with the three figures as sortable columns.
#[test]
fn the_results_list_groups_runs_by_comparable_heldout_set() {
    use crate::server::test_util::test_state_with;
    use askama::Template;
    let root = std::env::temp_dir().join(format!("bop-train-groups-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let done = |id: &str, person: Option<f64>, print: Option<&str>| {
        let mut r = record(
            id,
            SceneFilter::DAY,
            None,
            RunState::Done,
            person.map(|p| (p, 0.2, 0.1)),
        );
        if let Some(b) = r.benchmark.as_mut() {
            b.heldout_fingerprint = print.map(str::to_string);
        }
        r.output = Some(TrainOutput {
            name: r.name.clone(),
            onnx: root.join(format!("{id}.onnx")),
            yaml: root.join(format!("{id}.yaml")),
            provenance: root.join(format!("{id}.provenance.json")),
            family: ModelFamilyKind::Yolo26,
            classes: vec!["person".into()],
            license: "AGPL-3.0".into(),
            heldout: None,
            device: "mps".into(),
            train_seconds: 1.0,
        });
        r
    };
    let mut merged = done("g3", Some(0.9), Some("f1f1f1f1"));
    merged.request.rename = [("truck".to_string(), "car".to_string())].into();
    // Queue order, oldest first.
    let recs = vec![
        done("g1a", Some(0.5), Some("f1f1f1f1")),
        done("g1b", Some(0.7), Some("f1f1f1f1")),
        done("g2", Some(0.1), Some("f2abcdef99")),
        done("old", Some(0.3), None),
        done("nob", None, None),
        done("g1c", Some(0.6), Some("f1f1f1f1")),
        merged,
    ];
    let page = view_with(
        &state,
        MODELS,
        None,
        None,
        recs,
        crate::training::capability::host_for_tests(),
    )
    .render()
    .unwrap();
    let i = page.find("id=\"use\"").expect("the Use card");
    let card = &page[i..page[i..].find("</table>").map_or(page.len(), |e| i + e)];
    assert!(
        card.contains(
            "data-sort=\"results\" data-sort-default=\"best\" data-sort-cols=\",,,,,,,,-\""
        ),
        "{card}"
    );
    // The rows in page order: group heads and runs.
    let order: Vec<String> = card
        .split("<tr")
        .skip(1)
        .filter_map(|t| {
            if t.starts_with(" class=\"tr-group\" data-sort-group><td colspan=\"9\">") {
                let head = t
                    .split("<td colspan=\"9\">")
                    .nth(1)?
                    .split("</td>")
                    .next()?;
                Some(format!("# {head}"))
            } else if t.starts_with(" data-best=\"") {
                let best = t.split('"').nth(1)?;
                let name = t.split("<code>").nth(1)?.split("</code>").next()?;
                Some(format!("{best} {name}"))
            } else {
                None
            }
        })
        .collect();
    assert_eq!(
        order,
        [
            "# Held-out set f1f1f1\u{2026}, 1 run, classes person (merged: truck \u{2192} car)",
            "1 day-yolo26n-g3",
            "# Held-out set f1f1f1\u{2026}, 3 runs, classes person \u{2014} scores compare only \
             inside a group",
            "1 day-yolo26n-g1b",
            "2 day-yolo26n-g1c",
            "3 day-yolo26n-g1a",
            "# Not benchmarked",
            "1 day-yolo26n-nob",
            "# Held-out set not recorded (a run of an older version): its scores compare with \
             no other run",
            "1 day-yolo26n-old",
            "# Held-out set f2abcd\u{2026}, 1 run",
            "1 day-yolo26n-g2",
        ],
        "{card}"
    );
    // The figures beside the sentence; a dash without a benchmark.
    assert!(
        card.contains("<td class=\"num\" data-label=\"Person R\">0.70</td>"),
        "{card}"
    );
    assert!(card.contains("<td class=\"num\" data-label=\"Vehicle R\">\u{2014}</td>"));
    assert!(
        card.contains("<td class=\"num\" data-label=\"AP50-95\">0.333</td>"),
        "{card}"
    );
    // The detail rows follow their run and span every column.
    assert!(
        card.contains("<tr class=\"tr-switch-row\" data-sort-follow hidden><td colspan=\"9\">")
    );
    let _ = std::fs::remove_dir_all(&root);
}

#[tokio::test]
async fn the_page_shows_runs_comparisons_finish_and_the_new_fields() {
    use crate::server::test_util::{get_text, test_state_with};
    use askama::Template;
    let root = std::env::temp_dir().join(format!("bop-train-cmp-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(&root).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let mut recs = group_records();
    recs[2].output = Some(TrainOutput {
        name: recs[2].name.clone(),
        onnx: root.join("a.onnx"),
        yaml: root.join("a.yaml"),
        provenance: root.join("a.provenance.json"),
        family: ModelFamilyKind::Yolo26,
        classes: vec!["person".into()],
        license: "AGPL-3.0".into(),
        heldout: None,
        device: "mps".into(),
        train_seconds: 1.0,
    });
    let r9 = recs.iter_mut().find(|r| r.id == "r9").unwrap();
    r9.dataset = Some(crate::training::service::DatasetCounts {
        train_images: 412,
        balance_summary: Some("209 → 412 images/epoch; raccoon 32→118 boxes".into()),
        ..Default::default()
    });
    let page = view_with(
        &state,
        TRAIN,
        None,
        None,
        recs.clone(),
        crate::training::capability::host_for_tests(),
    )
    .render()
    .unwrap();
    // The run's dataset line carries its balance summary.
    let i = page.find("<tr data-run=\"r9\"").unwrap();
    let j = i + page[i..].find("</tr>").unwrap();
    assert!(
        page[i..j]
            .contains("412 train / 0 val / 0 held out \u{b7} balanced: 209 → 412 images/epoch"),
        "{}",
        &page[i..j]
    );
    // The Balance fieldset (collapsed, off by default) and the preview's holder.
    assert!(page.contains("<details class=\"tr-more tr-balance\" id=\"train-balance\">"));
    for a in ["class", "scene", "camera", "camera-day"] {
        assert!(
            page.contains(&format!("name=\"balance\" value=\"{a}\">")),
            "axis {a} unticked"
        );
    }
    assert!(page.contains("evens out day / night / IR frame counts"));
    assert!(page.contains("name=\"balance_strength\" value=\"0.5\" checked>"));
    assert!(page.contains("name=\"balance_strength\" value=\"1\">"));
    assert!(
        page.contains("name=\"balance_max_repeat\" min=\"1\" max=\"10\" step=\"1\" value=\"4\">")
    );
    assert!(page.contains("name=\"balance_cap_on\" id=\"train-bal-cap-on\">"));
    assert!(page.contains("name=\"balance_cap\" id=\"train-bal-cap\""));
    assert!(page.contains("id=\"train-balance-preview\""));
    // Finish now: shown for the running run in its training stage only.
    let row = |id: &str| {
        let i = page
            .find(&format!("<tr data-run=\"{id}\""))
            .unwrap_or_else(|| panic!("row {id}"));
        let j = i + page[i..].find("</tr>").unwrap();
        page[i..j].to_string()
    };
    let shown = "action=\"/training/finish\" class=\"inline\" data-field=\"finish\">";
    let hidden = "action=\"/training/finish\" class=\"inline\" data-field=\"finish\" hidden>";
    let (running, queued) = (row("r9"), row("r8"));
    assert!(running.contains(shown), "{running}");
    assert!(running.contains("Stop after this epoch and keep the best checkpoint"));
    // What Pause / Resume / Finish now do, for a finger as well as a mouse.
    assert!(
        running.contains("class=\"term term-q\" data-tip=\"Pause: its processes stop where they are;")
            && running.contains("Finish now: it stops after this epoch and keeps the best checkpoint.\" aria-label=\"What Pause, Resume and Finish now do\">?</button>"),
        "{running}"
    );
    assert!(
        queued.contains(hidden) && !queued.contains(shown),
        "{queued}"
    );
    let c = row("c");
    assert!(c.contains(hidden), "running but not training: {c}");
    assert_eq!(page.matches(shown).count(), 1);
    // Past an early-stop point: the notice, Stop now (the finish action) and Continue.
    assert!(
        running.contains(
            "data-field=\"extend\" data-run=\"r9\" data-ext=\"1\"><span data-field=\"extend-text\">Early stop reached at epoch 3"
        ),
        "{running}"
    );
    assert!(
        running.contains(">Stop now and keep epoch 2</button>"),
        "{running}"
    );
    assert!(running.contains("data-action=\"extend-continue\""));
    assert!(queued.contains("data-ext=\"0\" hidden>"), "{queued}");
    // The Comparison card, on the Models tab: the newest group open, the routed row, the best
    // cells.
    let models = view_with(
        &state,
        MODELS,
        None,
        None,
        recs,
        crate::training::capability::host_for_tests(),
    )
    .render()
    .unwrap();
    let i = models.find("id=\"compare\"").expect("the Comparison card");
    let card = &models[i..];
    assert!(
        card.contains("<details class=\"tr-compare\" open>"),
        "{card}"
    );
    assert!(card.find("cmp-new").unwrap() < card.find("cmp-old").unwrap());
    assert!(card.contains("day-yolo26n-a"));
    assert!(card.contains("class=\"tr-routed\""));
    assert!(card.contains("day + night+ir routed"));
    assert!(card.contains("class=\"num best-val\""));
    assert!(!card.contains("cmp-solo"));
    // The column help and each cell's detail open on a tap too (a title shows on hover only).
    assert!(card.contains(
        "<th class=\"num\"><button type=\"button\" class=\"term\" data-tip=\"Recall of person on every held-out frame at the confidence threshold the model runs with (IoU 0.5)\">Person R</button></th>"
    ));
    assert!(card.contains(">False alerts</button></th>"));
    assert!(!card.contains("<th class=\"num\" title="), "{card}");
    assert!(
        card.contains("<td class=\"num best-val\"><button type=\"button\" class=\"term\" data-tip=\"2 of 40 frames without any object alerted (5.0%) at threshold 0.50\">2/40</button></td>"),
        "{card}"
    );
    assert!(card.contains("held-out images\">0.600</button>"), "{card}");
    // The Runs checkboxes, Patience, the recommendation and the class picker's fields.
    assert!(page.contains("name=\"scene\" value=\"all\" data-prefix=\"custom\" checked>"));
    assert!(page.contains("name=\"scene\" value=\"night+ir\" data-prefix=\"night\">"));
    assert!(page.contains("name=\"scene\" value=\"ir\" data-prefix=\"ir\">"));
    assert!(page.contains("name=\"patience\""));
    assert!(
        page.contains(
            "name=\"epochs\" min=\"1\" max=\"1000\" value=\"\" placeholder=\"recommended\""
        )
    );
    assert!(page.contains("100 ep \u{b7} patience 15"));
    assert!(page.contains("80 ep \u{b7} patience 10"));
    // The learning rate: the family defaults in the recommendation, on every size checkbox
    // (the placeholder follows the ticked family) and in the hints.
    assert!(page.contains("100 ep \u{b7} patience 15 \u{b7} lr auto (optimizer picks)"));
    assert!(page.contains("lr 1e-4 (5e-5 under 1000 unique frames)"));
    assert!(page.contains(
        "value=\"yolo26n\" aria-label=\"yolo26n\" data-default-lr=\"auto (optimizer picks)\""
    ));
    // No inputmode: a decimal keypad could not type `1e-4`.
    assert!(page.contains("name=\"lr\" autocomplete=\"off\""));
    assert!(!page.contains("name=\"lr\" inputmode"));
    assert!(page.contains("placeholder=\"recommended\""));
    assert!(page.contains("RF-DETR: cosine (default) or step, final factor 0.1"));
    assert!(page.contains("<option value=\"\" selected>default</option>"));
    assert!(page.contains("name=\"lr_min_factor\""));
    // Select best by: the one list, with hints; mAP preselected.
    assert!(page.contains("name=\"select_metric\""));
    for o in recipe::SELECT_METRICS {
        assert!(page.contains(&format!("<option value=\"{}\"", o.metric.id())));
        assert!(page.contains(o.label), "{}", o.label);
    }
    assert!(page.contains("ignores missed objects"));
    assert!(page.contains("<option value=\"map\" data-hint"));
    assert!(page.contains("held-out benchmark is the number to trust"));
    assert!(page.contains("id=\"train-picker\""));
    assert!(page.contains("id=\"train-merges\""));
    assert!(page.contains("name=\"rename\""));
    assert!(page.contains("name=\"excluded\" value=\"drop-frame\""));
    // The per-run summary of a finished run: AP50-95 per scene and in scope.
    assert!(
        models.contains("AP50-95 day 0.600 / night 0.200 / IR 0.100 \u{b7} in scope 0.600"),
        "the result row"
    );
    // The live page renders too, and the compare JSON answers 400 / 404.
    let (s, _) = get_text(&state, "/train/models").await;
    assert_eq!(s, StatusCode::OK);
    let (s, text) = get_text(&state, "/v1/training/compare").await;
    assert_eq!(s, StatusCode::BAD_REQUEST, "{text}");
    let (s, text) = get_text(&state, "/v1/training/compare?group=cmp-nope").await;
    assert_eq!(s, StatusCode::NOT_FOUND, "{text}");
    assert!(text.contains("cmp-nope"));
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn a_trained_model_links_to_compare_on_its_held_out_images() {
    let b: BenchSummary = serde_json::from_value(serde_json::json!({
        "datasets": ["heldout-a b", "coco-cctv"],
        "device": null,
        "ms_p50": null,
        "ap50": null,
        "ap50_95": null,
        "images": 0,
    }))
    .unwrap();
    assert_eq!(
        compare_href(Some(&b)),
        "/benchmark/compare?set=heldout-a%20b"
    );
    assert_eq!(compare_href(None), "/benchmark/compare");
}

#[test]
fn bench_text_reads_well() {
    let b = BenchSummary {
        datasets: vec!["heldout-x".into()],
        device: Some("openvino:cpu".into()),
        ms_p50: Some(12.34),
        ap50: Some(0.5),
        ap50_95: None,
        images: 20,
        ..BenchSummary::default()
    };
    assert_eq!(
        bench_text(Some(&b), None),
        "AP50 0.500 \u{b7} 12.3 ms on openvino:cpu \u{b7} 20 images (heldout-x)"
    );
    assert_eq!(bench_text(None, Some("x")), "benchmark failed: x");
    // The classes the AP averages, when recorded.
    let mut based = b.clone();
    based.ap_basis = Some(crate::benchmark::metrics::ApBasis {
        min_gt: 20,
        classes: vec!["car".into(), "person".into()],
        fell_back: false,
    });
    assert!(
        bench_text(Some(&based), None).starts_with("AP50 0.500 (car, person) \u{b7} 12.3 ms"),
        "{}",
        bench_text(Some(&based), None)
    );
    based.ap_basis.as_mut().unwrap().fell_back = true;
    assert!(
        bench_text(Some(&based), None)
            .starts_with("AP50 0.500 (every class: none had 20 objects) \u{b7}"),
        "{}",
        bench_text(Some(&based), None)
    );
    // Per scene (only scenes with held-out frames) and in scope.
    let mut b = b;
    b.by_scene.insert("day".into(), scene_ap(0.61));
    b.by_scene.insert("ir".into(), scene_ap(0.3));
    b.by_scene.insert(
        "night".into(),
        crate::training::job::SceneAp {
            images: 0,
            ..scene_ap(0.1)
        },
    );
    b.in_scope = Some(scene_ap(0.55));
    assert!(
        bench_text(Some(&b), None)
            .ends_with(" \u{b7} AP50-95 day 0.610 / IR 0.300 \u{b7} in scope 0.550"),
        "{}",
        bench_text(Some(&b), None)
    );
    // Alerting at the threshold: only the parts present.
    b.threshold = Some(0.5);
    b.person_recall = Some(0.391);
    assert!(
        bench_text(Some(&b), None)
            .ends_with(" \u{b7} in scope 0.550 \u{b7} person recall 0.39 @ 0.50"),
        "{}",
        bench_text(Some(&b), None)
    );
    b.vehicle_recall = Some(0.666);
    b.empty_frames = Some(130);
    b.empty_frame_alerts = Some(0);
    assert!(
        bench_text(Some(&b), None).ends_with(
            " \u{b7} person recall 0.39 \u{b7} vehicle recall 0.67 \u{b7} false alerts 0/130 empty \
             frames @ 0.50"
        ),
        "{}",
        bench_text(Some(&b), None)
    );
    // False person alerts on frames without a person (a parked car does not hide them).
    b.person_free_frames = Some(210);
    b.person_false_alerts = Some(3);
    assert!(
        bench_text(Some(&b), None).ends_with(
            "false alerts 0/130 empty frames \u{b7} person alerts 3/210 frames without a person \
             @ 0.50"
        ),
        "{}",
        bench_text(Some(&b), None)
    );
    b.person_free_frames = None;
    // No empty frames: no false-alert part.
    b.empty_frames = Some(0);
    assert!(bench_text(Some(&b), None).ends_with("vehicle recall 0.67 @ 0.50"));
}

#[test]
fn balance_fields_parse_default_from_the_config_and_name_bad_values() {
    use crate::training::config::TrainingConfig;
    use std::collections::BTreeSet;
    let t = TrainingConfig::default();
    let axes = |v: &[BalanceAxis]| v.iter().copied().collect::<BTreeSet<_>>();
    // Repeated checkboxes and a comma list, strength, max repeat and the gated cap.
    let b = balance_of(
        &form(&[
            ("balance", "class"),
            ("balance", "scene, camera-day"),
            ("balance_strength", "1"),
            ("balance_max_repeat", "6"),
            ("balance_cap_on", "on"),
            ("balance_cap", "2.5"),
        ]),
        &t,
    )
    .unwrap();
    assert_eq!(
        b.axes,
        axes(&[
            BalanceAxis::Class,
            BalanceAxis::Scene,
            BalanceAxis::CameraDay
        ])
    );
    assert_eq!((b.strength, b.max_repeat), (1.0, 6));
    assert_eq!(b.cap_dominant, Some(2.5));
    // The cap applies only with its box; a blank cap is the default 3.
    let b = balance_of(&form(&[("balance", "camera"), ("balance_cap", "2")]), &t).unwrap();
    assert_eq!(b.cap_dominant, None);
    assert_eq!((b.strength, b.max_repeat), (0.5, 4), "defaults");
    let b = balance_of(
        &form(&[
            ("balance", "camera"),
            ("balance_cap_on", "on"),
            ("balance_cap", ""),
        ]),
        &t,
    )
    .unwrap();
    assert_eq!(b.cap_dominant, Some(3.0));
    // No balance field at all: training.balance, else off.
    let configured = TrainingConfig {
        balance: Some(Balance {
            axes: axes(&[BalanceAxis::Class]),
            strength: 1.0,
            ..Balance::default()
        }),
        ..TrainingConfig::default()
    };
    let none = form(&[("dataset", "/x")]);
    assert_eq!(
        &balance_of(&none, &configured).unwrap(),
        configured.balance.as_ref().unwrap()
    );
    assert!(balance_of(&none, &t).unwrap().is_off());
    // A form with the fieldset but no axis ticked: off, whatever the config or the other
    // fields say.
    let off = form(&[("balance_strength", "7"), ("balance_max_repeat", "4")]);
    assert_eq!(balance_of(&off, &configured).unwrap(), Balance::default());
    // Bad values name the field.
    let err = |pairs: &[(&str, &str)]| {
        let mut v = vec![("balance", "class")];
        v.extend_from_slice(pairs);
        format!("{:#}", balance_of(&form(&v), &t).unwrap_err())
    };
    let e = format!(
        "{:#}",
        balance_of(&form(&[("balance", "class,colour")]), &t).unwrap_err()
    );
    assert!(e.contains("balance") && e.contains("'colour'"), "{e}");
    let e = err(&[("balance_strength", "lots")]);
    assert!(
        e.contains("balance strength") && e.contains("'lots'"),
        "{e}"
    );
    assert!(err(&[("balance_strength", "1.5")]).contains("balance.strength"));
    assert!(err(&[("balance_strength", "NaN")]).contains("balance.strength"));
    assert!(err(&[("balance_max_repeat", "0")]).contains("balance.max_repeat"));
    assert!(err(&[("balance_max_repeat", "11")]).contains("balance.max_repeat"));
    assert!(err(&[("balance_max_repeat", "-1")]).contains("balance max repeat"));
    assert!(err(&[("balance_cap_on", "on"), ("balance_cap", "0.5")]).contains("cap_dominant"));
    // Every run of a scenes x models batch gets the same balance.
    let (reqs, _) = requests_from_form(
        &form(&[
            ("model", "yolo26n"),
            ("model", "yolo26s"),
            ("scene", "day"),
            ("scene", "night+ir"),
            ("balance", "class,scene"),
            ("balance_strength", "0.5"),
            ("balance_max_repeat", "5"),
        ]),
        &mac(),
        &t,
    )
    .unwrap();
    assert_eq!(reqs.len(), 4);
    for r in &reqs {
        assert_eq!(
            r.balance.axes,
            axes(&[BalanceAxis::Class, BalanceAxis::Scene])
        );
        assert_eq!(r.balance.max_repeat, 5);
    }
    // A bad balance refuses the queue form.
    let e = requests_from_form(
        &form(&[
            ("model", "yolo26n"),
            ("balance", "class"),
            ("balance_max_repeat", "0"),
        ]),
        &mac(),
        &t,
    )
    .unwrap_err();
    assert!(format!("{e:#}").contains("balance.max_repeat"), "{e:#}");
}

/// A review export of 3 cameras x (9 day + 3 IR frames), every fourth one empty, a
/// raccoon in the first IR frame of each camera, persons elsewhere.
fn skewed_export(dir: &std::path::Path) {
    std::fs::create_dir_all(dir.join("images")).unwrap();
    std::fs::create_dir_all(dir.join("labels")).unwrap();
    std::fs::write(dir.join("classes.txt"), "person\nraccoon\n").unwrap();
    for cam in ["a", "b", "c"] {
        for (scene, date, n) in [("day", "2026-10-01", 9), ("ir", "2026-10-02", 3)] {
            for i in 0..n {
                let stem = format!("{scene}.{cam}.{date}.f{i}");
                std::fs::write(dir.join("images").join(format!("{stem}.jpg")), b"x").unwrap();
                let label = match (scene, i % 4) {
                    (_, 3) => "",
                    ("ir", 0) => "1 0.5 0.5 0.2 0.2\n",
                    _ => "0 0.5 0.5 0.2 0.4\n",
                };
                std::fs::write(dir.join("labels").join(format!("{stem}.txt")), label).unwrap();
            }
        }
    }
}

#[tokio::test]
async fn the_balance_preview_shows_before_and_after_per_run_and_checks_folders() {
    use crate::server::test_util::{REMOTE, call, from_peer, post_form, test_state_with};
    let root = std::env::temp_dir().join(format!("bop-train-bal-{}", uuid::Uuid::new_v4()));
    let export = root.join("exports/review-1");
    skewed_export(&export);
    let (r2, e2) = (root.clone(), export.clone());
    let state = test_state_with(move |c| {
        c.download_dir = Some(r2);
        c.training.datasets = vec![e2];
    });
    let enc = super::super::encode_uri_component(&export.display().to_string());
    let post = |body: String| post_form("/v1/training/balance-preview", &body, false);
    let json = |b: &[u8]| -> serde_json::Value { serde_json::from_slice(b).unwrap() };
    // Class balance over two runs (all scenes, day): the raccoon frames repeat.
    let (s, _, b) = call(
        &state,
        post(format!(
            "dataset={enc}&scene=all&scene=day&balance=class&balance_strength=1&\
             balance_max_repeat=4"
        )),
    )
    .await;
    let v = json(&b);
    assert_eq!(s, StatusCode::OK, "{v}");
    assert_eq!(v["success"], true);
    assert_eq!(v["balance"]["axes"], serde_json::json!(["class"]));
    let runs = v["runs"].as_array().unwrap();
    assert_eq!(runs.len(), 2);
    assert_eq!(runs[0]["scene"], "all");
    assert_eq!(runs[0]["title"], "All scenes (mixed)");
    assert_eq!(runs[1]["scene"], "day");
    let r = &runs[0];
    assert!(r["error"].is_null(), "{r}");
    let rep = &r["report"];
    let before = rep["before"]["boxes_per_class"]["raccoon"]
        .as_u64()
        .unwrap();
    let after = rep["after"]["boxes_per_class"]["raccoon"].as_u64().unwrap();
    assert!(before > 0 && after > before, "{rep}");
    assert_eq!(
        rep["before"]["boxes_per_class"]["person"],
        rep["after"]["boxes_per_class"]["person"]
    );
    assert!(rep["after"]["images"].as_u64() > rep["before"]["images"].as_u64());
    assert_eq!(r["train_images"], rep["after"]["images"]);
    assert!(r["train_frames"].as_u64().unwrap() > 0);
    assert!(r["heldout_frames"].as_u64().unwrap() > 0, "{r}");
    assert!(r["warnings"].is_array());
    // The pre-flight report and the disk come with every answer.
    let pre = &r["preflight"];
    assert_eq!(pre["train_frames"], r["train_frames"]);
    assert_eq!(pre["heldout_frames"], r["heldout_frames"]);
    assert!(
        pre["classes"]
            .as_array()
            .unwrap()
            .iter()
            .any(|c| c["name"] == "raccoon")
    );
    assert!(pre["refusals"].as_array().unwrap().is_empty(), "{pre}");
    assert!(
        v["disk"]["lines"].as_array().is_some_and(|l| !l.is_empty()),
        "{v}"
    );
    assert!(
        v["disk"]["lines"][0]["text"]
            .as_str()
            .unwrap()
            .contains("needed under")
    );
    // Balance off: no report, the counts still.
    let (s, _, b) = call(&state, post(format!("dataset={enc}&balance_strength=0.5"))).await;
    let v = json(&b);
    assert_eq!(s, StatusCode::OK, "{v}");
    let r = &v["runs"][0];
    assert!(r["report"].is_null(), "{r}");
    assert_eq!(r["train_images"], r["train_frames"]);
    // A folder outside the offered and browsable ones, a bad field: 400 naming it.
    let (s, _, b) = call(
        &state,
        post("dataset_path=%2Fno%2Fsuch%2Fexport&balance=class".into()),
    )
    .await;
    let v = json(&b);
    assert_eq!(s, StatusCode::BAD_REQUEST, "{v}");
    assert_eq!(v["success"], false);
    assert!(
        v["message"].as_str().unwrap().contains("another folder"),
        "{v}"
    );
    let (s, _, b) = call(&state, post(format!("dataset={enc}&balance=colour"))).await;
    assert_eq!(s, StatusCode::BAD_REQUEST);
    assert!(json(&b)["message"].as_str().unwrap().contains("'colour'"));
    // A class the datasets do not have: that run's error.
    let (_, _, b) = call(
        &state,
        post(format!("dataset={enc}&classes=zebra&balance=class")),
    )
    .await;
    let v = json(&b);
    assert!(
        v["runs"][0]["error"]
            .as_str()
            .is_some_and(|e| !e.is_empty()),
        "{v}"
    );
    // Cross-site browser requests and requests that are not signed in are refused.
    let mut req = post(format!("dataset={enc}&balance=class"));
    req.headers_mut()
        .insert("sec-fetch-site", "cross-site".parse().unwrap());
    let (s, _, _) = call(&state, req).await;
    assert_eq!(s, StatusCode::FORBIDDEN);
    let (s, _, _) = call(
        &state,
        crate::server::test_util::anonymous(from_peer(post(format!("dataset={enc}")), REMOTE)),
    )
    .await;
    assert_eq!(s, StatusCode::UNAUTHORIZED);
    let _ = std::fs::remove_dir_all(&root);
}

#[test]
fn the_balance_fieldset_starts_from_training_balance() {
    use askama::Template;
    let state = crate::server::test_util::test_state_with(|c| {
        c.training.balance = Some(Balance {
            axes: Balance::parse_axes("scene,camera-day").unwrap(),
            strength: 0.7,
            max_repeat: 6,
            cap_dominant: Some(2.5),
        });
    });
    let page = view_with(
        &state,
        TRAIN,
        None,
        None,
        Vec::new(),
        crate::training::capability::host_for_tests(),
    )
    .render()
    .unwrap();
    assert!(page.contains("<details class=\"tr-more tr-balance\" id=\"train-balance\" open>"));
    assert!(page.contains("name=\"balance\" value=\"scene\" checked>"));
    assert!(page.contains("name=\"balance\" value=\"camera-day\" checked>"));
    assert!(page.contains("name=\"balance\" value=\"class\">"));
    // A configured strength that is neither preset: its own radio, checked.
    assert!(page.contains("name=\"balance_strength\" value=\"0.5\">"));
    assert!(page.contains("name=\"balance_strength\" value=\"0.7\" checked> Configured (0.7)"));
    assert!(page.contains("value=\"6\">"));
    assert!(page.contains("name=\"balance_cap_on\" id=\"train-bal-cap-on\" checked>"));
    assert!(page.contains("value=\"2.5\""));
}

/// A task that ends at once, or (`hold`) runs until cancelled.
struct EnvTask {
    id: &'static str,
    hold: bool,
}

impl crate::training::task::Task for EnvTask {
    fn id(&self) -> &str {
        self.id
    }
    fn kind(&self) -> crate::training::task::TaskKind {
        crate::training::task::TaskKind::Import
    }
    fn title(&self) -> String {
        self.id.into()
    }
    fn steps(&self) -> usize {
        1
    }
    fn run(
        &self,
        _ctx: crate::training::task::TaskContext<'_>,
        _report: &mut dyn FnMut(crate::training::task::TaskReport<'_>),
        cancel: &AtomicBool,
    ) -> anyhow::Result<crate::training::task::TaskOutcome> {
        while self.hold && !cancel.load(std::sync::atomic::Ordering::Relaxed) {
            std::thread::sleep(std::time::Duration::from_millis(5));
        }
        Ok(crate::training::task::TaskOutcome {
            summary: "done".into(),
            result: serde_json::Value::Null,
        })
    }
}

fn env_queue() -> Arc<TrainingService> {
    let fake = Arc::new(crate::resources::export::fake::Fake::new());
    TrainingService::start(
        fake.clone(),
        fake,
        Arc::new(crate::training::job::NoHooks),
        None,
    )
}

/// A training environment on disk under a fresh root.
fn env_on_disk(tag: &str) -> (PathBuf, TrainPaths) {
    let root = std::env::temp_dir().join(format!("bop-remove-env-{tag}-{}", uuid::Uuid::new_v4()));
    let paths = TrainPaths::new(&root, root.join("models"));
    std::fs::create_dir_all(&paths.export.env).unwrap();
    std::fs::write(paths.export.env.join("pyvenv.cfg"), b"home = x").unwrap();
    (root, paths)
}

/// Audit 2026-11 (QA-021): Remove training environment and the training folder's save run
/// with every enqueue held off: a job queued from another request while the action runs
/// waits until it returns, so no job starts on an environment being deleted or is queued
/// into the old folder.
#[test]
fn an_enqueue_waits_for_the_idle_action() {
    let q = env_queue();
    let entered = Arc::new(std::sync::Barrier::new(2));
    let release = Arc::new(std::sync::Barrier::new(2));
    let queued = Arc::new(AtomicBool::new(false));
    let action = {
        let (q, entered, release) = (q.clone(), entered.clone(), release.clone());
        std::thread::spawn(move || {
            while_queue_idle(Some(&q), || {
                entered.wait();
                release.wait();
            })
        })
    };
    entered.wait();
    let enqueue = {
        let (q, queued) = (q.clone(), queued.clone());
        std::thread::spawn(move || {
            q.enqueue_task(Box::new(EnvTask {
                id: "after",
                hold: false,
            }))
            .unwrap();
            queued.store(true, std::sync::atomic::Ordering::SeqCst);
        })
    };
    std::thread::sleep(std::time::Duration::from_millis(300));
    assert!(
        !queued.load(std::sync::atomic::Ordering::SeqCst),
        "the enqueue waits for the action"
    );
    release.wait();
    action.join().unwrap().unwrap();
    enqueue.join().unwrap();
    assert!(queued.load(std::sync::atomic::Ordering::SeqCst));
    // A busy queue refuses the action without running it.
    q.enqueue_task(Box::new(EnvTask {
        id: "holds",
        hold: true,
    }))
    .unwrap();
    let ran = AtomicBool::new(false);
    let deadline = std::time::Instant::now() + std::time::Duration::from_secs(10);
    while !q.busy() {
        assert!(std::time::Instant::now() < deadline);
        std::thread::sleep(std::time::Duration::from_millis(5));
    }
    assert!(
        while_queue_idle(Some(&q), || ran
            .store(true, std::sync::atomic::Ordering::SeqCst))
        .is_err()
    );
    assert!(!ran.load(std::sync::atomic::Ordering::SeqCst));
    q.cancel("holds");
    q.shutdown();
}

/// Audit 2026-11 (RA-052, QA-021): a training, import or auto-label of another process (here
/// another handle) holds `tools/.train.lock`: 409 and nothing removed, with or without a
/// queue in this process. Once free, the environment goes and the lock file stays.
#[test]
fn remove_env_refuses_a_held_training_lock() {
    let (root, paths) = env_on_disk("locked");
    let q = env_queue();
    let held = env::TrainLock::try_acquire(&paths.train_lock_file()).unwrap();
    for queue in [None, Some(&*q)] {
        let (code, message) = remove_env_in(queue, &paths).unwrap_err();
        assert_eq!(code, StatusCode::CONFLICT, "{message}");
        assert!(message.contains("wait for it to finish"), "{message}");
        assert!(
            paths.export.env.join("pyvenv.cfg").exists(),
            "nothing removed"
        );
    }
    drop(held);
    assert_eq!(
        remove_env_in(Some(&q), &paths).unwrap(),
        "Removed the training environment."
    );
    assert!(!paths.export.env.exists());
    assert!(paths.train_lock_file().is_file(), "the lock file stays");
    q.shutdown();
    std::fs::remove_dir_all(&root).ok();
}

/// A job queued or running in this process: 409 with the page's sentence, nothing removed.
#[test]
fn remove_env_refuses_a_busy_queue() {
    let (root, paths) = env_on_disk("busy");
    let q = env_queue();
    q.enqueue_task(Box::new(EnvTask {
        id: "holds",
        hold: true,
    }))
    .unwrap();
    let (code, message) = remove_env_in(Some(&q), &paths).unwrap_err();
    assert_eq!(code, StatusCode::CONFLICT);
    assert_eq!(message, "A run is queued or running; cancel it first.");
    assert!(paths.export.env.exists());
    q.cancel("holds");
    q.shutdown();
    std::fs::remove_dir_all(&root).ok();
}

/// QA-243: the routes themselves, not only [`remove_env_in`]. While a job of this process's queue
/// runs, `POST /training/remove-env` and `POST /training/dir` answer 409 and change nothing; while
/// another process (here another handle) holds `tools/.train.lock`, remove-env answers 409 and
/// removes nothing. Once both are gone, remove-env deletes the environment (200, the lock file
/// stays) and the training folder saves.
#[tokio::test]
async fn remove_env_and_training_dir_routes_refuse_while_busy() {
    use crate::server::test_util::{call, post_form, test_state_with};
    let root = std::env::temp_dir().join(format!("bop-remove-env-route-{}", uuid::Uuid::new_v4()));
    std::fs::create_dir_all(root.join("my-runs")).unwrap();
    let r2 = root.clone();
    let state = test_state_with(move |c| c.download_dir = Some(r2));
    let paths = TrainPaths::for_config(&state.config_read());
    std::fs::create_dir_all(&paths.export.env).unwrap();
    std::fs::write(paths.export.env.join("pyvenv.cfg"), b"home = x").unwrap();
    let html = |b: Vec<u8>| String::from_utf8(b).unwrap();
    async fn busy_until(want: bool) {
        let deadline = std::time::Instant::now() + std::time::Duration::from_secs(120);
        while started_queue().is_some_and(|s| s.busy()) != want {
            assert!(std::time::Instant::now() < deadline, "queue busy != {want}");
            tokio::time::sleep(std::time::Duration::from_millis(5)).await;
        }
    }

    let _queue = own_queue(&state).await;
    let svc = service(&Shared::from(state.clone()));
    let id = "remove-env-route-holds";
    svc.enqueue_task(Box::new(EnvTask { id, hold: true }))
        .unwrap();
    busy_until(true).await;
    let (s, _, b) = call(&state, post_form("/training/remove-env", "", true)).await;
    let page = html(b);
    assert_eq!(s, StatusCode::CONFLICT, "{page}");
    assert!(
        page.contains("A run is queued or running; cancel it first."),
        "{page}"
    );
    assert!(
        paths.export.env.join("pyvenv.cfg").exists(),
        "nothing removed"
    );
    let (s, _, b) = call(
        &state,
        post_form("/training/dir", "training_dir=my-runs", true),
    )
    .await;
    let page = html(b);
    assert_eq!(s, StatusCode::CONFLICT, "{page}");
    assert!(
        page.contains("Not saved: a training run, import or auto-label job is queued or running"),
        "{page}"
    );
    assert_eq!(state.config_read().training.dir, None, "nothing saved");
    assert!(svc.cancel(id));
    busy_until(false).await;

    // A training of another process holds the tool lock.
    let held = env::TrainLock::try_acquire(&paths.train_lock_file()).unwrap();
    let (s, _, b) = call(&state, post_form("/training/remove-env", "", true)).await;
    let page = html(b);
    assert_eq!(s, StatusCode::CONFLICT, "{page}");
    assert!(page.contains("wait for it to finish"), "{page}");
    assert!(
        paths.export.env.join("pyvenv.cfg").exists(),
        "nothing removed"
    );
    drop(held);

    let (s, _, b) = call(&state, post_form("/training/remove-env", "", true)).await;
    let page = html(b);
    assert_eq!(s, StatusCode::OK, "{page}");
    assert!(page.contains("Removed the training environment."), "{page}");
    assert!(!paths.export.env.exists());
    assert!(paths.train_lock_file().is_file(), "the lock file stays");
    let (s, _, b) = call(
        &state,
        post_form("/training/dir", "training_dir=my-runs", true),
    )
    .await;
    let page = html(b);
    assert_eq!(s, StatusCode::OK, "{page}");
    assert!(page.contains("stay there (not moved)"), "{page}");
    assert!(state.config_read().training.dir.is_some());
    let (s, _, _) = call(&state, post_form("/training/dir", "training_dir=", true)).await;
    assert_eq!(s, StatusCode::OK);
    assert_eq!(state.config_read().training.dir, None);
    let _ = std::fs::remove_dir_all(&root);
}
