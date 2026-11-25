//! Characterisation of [`run_training`]'s observable output (QA-135): every report, the
//! arguments of every `train_model.py` command and the run log, for a run that restarts after
//! running out of GPU memory (with a Finish request carried over), one that restarts after running
//! out of system memory, and an export retry. Pinned before the stage machine was split into
//! stage functions; a difference here is a change of behaviour.

use super::*;

/// `s` with the test's folder and run id replaced, `/` separators, and what depends on the
/// clock left out: the time left (`, about 3 s left`) and the time of a self-check that passed.
fn norm(s: &str, root: &Path, id: &str) -> String {
    let mut s = s
        .replace(&root.display().to_string(), "<T>")
        .replace(id, "<ID>")
        .replace('\\', "/");
    for (start, end, with) in [
        (", about ", " left", ""),
        (
            "(unchanged since ",
            ": the same",
            "(unchanged since <TIME>: the same",
        ),
    ] {
        let mut from = 0;
        while let Some(a) = s[from..].find(start).map(|a| a + from) {
            let Some(b) = s[a..].find(end) else { break };
            s.replace_range(a..a + b + end.len(), with);
            from = a + with.len();
        }
    }
    s
}

/// The transcript of running `j` on `fake`: the outcome, each report, each training command's
/// arguments and the log (its version lines and resource lines, host readings, left out).
fn transcript(j: &TrainJob, fake: &TrainFake, hooks: &dyn Hooks, root: &Path) -> String {
    let n = |s: &str| norm(s, root, &j.id);
    let mut out: Vec<String> = Vec::new();
    let mut settings = None;
    let r = run_training(
        j,
        fake,
        fake,
        hooks,
        &mut |p| {
            match p {
            TrainReport::Stage {
                stage,
                percent,
                detail,
            } => out.push(format!("stage {stage:?} {percent} {}", n(detail))),
            TrainReport::Line(l) => out.push(format!("line {}", n(l))),
            TrainReport::Progress(tp) => {
                out.push(format!(
                    "progress {} | batch {:?} | note {:?} | stop {} {} {} {}",
                    n(&tp.describe()),
                    tp.batch_size,
                    tp.note,
                    tp.patience,
                    tp.min_delta,
                    tp.min_epochs,
                    tp.smooth
                ));
                let s = format!("{:?}", tp.settings);
                if settings.as_ref() != Some(&s) {
                    out.push(format!("settings {s}"));
                    settings = Some(s);
                }
            }
            TrainReport::Prepared(p) => out.push(format!(
                "prepared {} train ({} frames), {} valid, {} heldout, classes {:?}, heldout dir {:?}",
                p.train_images,
                p.train_frames,
                p.valid_images,
                p.heldout_images,
                p.classes,
                p.heldout_dir.as_ref().map(|d| n(&d.display().to_string()))
            )),
            TrainReport::Batch(b) => out.push(format!("batch {b}")),
            TrainReport::Installed(o) => out.push(format!(
                "installed {} {} {:?} {}",
                o.name,
                n(&o.onnx.display().to_string()),
                o.classes,
                o.device
            )),
            TrainReport::Registered(m) => out.push(format!("registered {m}")),
            TrainReport::Resources(r) => {
                out.push(format!(
                    "resources workers {} loaders {} forced {} limit {:?} host_oom {:?} {:?} img_s {:?}",
                    r.workers, r.loaders, r.forced, r.limit, r.host_oom, r.host_oom_img_s, r.img_s
                ));
            }
        }
        },
        &AtomicBool::new(false),
    );
    out.push(match r {
        Ok(o) => format!(
            "ok {} registered {:?} benchmark {:?} error {:?}",
            o.output.name, o.registered, o.benchmark, o.benchmark_error
        ),
        Err(e) => format!(
            "err {:?} {} cancelled {}",
            e.stage,
            n(&e.message),
            e.cancelled
        ),
    });
    for c in fake.train_cmds.lock().unwrap().iter() {
        let args: Vec<String> = c.args.iter().map(|a| n(&a.to_string_lossy())).collect();
        out.push(format!("cmd {:?} {}", c.stage, args.join(" ")));
    }
    let log = std::fs::read_to_string(j.run_dir.join(LOG_FILE)).unwrap_or_default();
    let version = format!("# {} ", crate::long_version());
    for l in log.lines() {
        if !l.starts_with(&version) && !l.starts_with("# resources: ") {
            out.push(format!("log {}", n(l)));
        }
    }
    let mut s = out.join("\n");
    s.push('\n');
    s
}

/// Asserts `got` equals `want` line by line (the first difference named).
fn same(got: &str, want: &str) {
    for (i, (g, w)) in got.lines().zip(want.lines()).enumerate() {
        assert_eq!(g, w, "line {} differs; whole transcript:\n{got}", i + 1);
    }
    assert_eq!(
        got.lines().count(),
        want.lines().count(),
        "whole transcript:\n{got}"
    );
}

#[test]
fn a_gpu_out_of_memory_run_with_a_finish_request() {
    let t = tmp("transcript-gpu");
    let mut j = job(
        &t.0,
        TrainFamily::Rfdetr,
        "nano",
        mps(),
        Platform::MACOS_ARM64,
    );
    j.request.recipe.batch = 16;
    let fake = TrainFake {
        oom_above: Some(4),
        oom_epochs: 1,
        finish_before_oom: true,
        ..TrainFake::new()
    };
    let got = transcript(&j, &fake, &RecHooks::default(), &t.0);
    same(&got, GPU_OOM);
}

#[test]
fn a_host_out_of_memory_run() {
    let t = tmp("transcript-host");
    let mut j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    j.workers = Workers::Fixed(4);
    let fake = TrainFake {
        host_oom_above: Some(1),
        ..TrainFake::new()
    };
    let got = transcript(&j, &fake, &NoHooks, &t.0);
    same(&got, HOST_OOM_RUN);
}

#[test]
fn an_export_retry() {
    let t = tmp("transcript-retry");
    let j = job(&t.0, TrainFamily::Yolo26, "n", mps(), Platform::MACOS_ARM64);
    let fake = TrainFake {
        checkpoint: true,
        export_fails: vec!["RuntimeError: ONNX Runtime and PyTorch disagree"],
        ..TrainFake::new()
    };
    let first = transcript(&j, &fake, &NoHooks, &t.0);
    let again = retry_of(&j);
    let fake = TrainFake::new();
    let got = transcript(&again, &fake, &RecHooks::default(), &t.0);
    same(&first, FAILED_EXPORT);
    same(&got, EXPORT_RETRY);
}

const GPU_OOM: &str = "\
stage Setup 0 checking the toolchain
stage Setup 0 downloading uv: checking the toolchain
stage Setup 0 downloading uv: uv (model export toolchain)
stage Setup 2 downloading uv: 7.1 of 14.2 MB
stage Setup 5 installing Python 3.11: uv python install 3.11
stage Setup 11 installing Python 3.11: uv venv
stage Setup 16 installing the pinned Python packages: train-macos-arm64.txt
line Resolved 48 packages in 3ms
line Downloading torch (75.3MiB)
stage Setup 16 installing the pinned Python packages: downloading torch (75.3MiB)
line  Downloaded torch
stage Setup 85 installing the pinned Python packages: downloaded torch
line Prepared 48 packages in 40s
stage Setup 90 installing the pinned Python packages: Prepared 48 packages in 40s
line Installed 48 packages in 2s
stage Setup 98 installing the pinned Python packages: Installed 48 packages in 2s
stage Setup 98 installing the pinned Python packages: packages installed
stage Setup 100 checking the Visual C++ runtime: not needed on macos
stage SelfCheck 0 importing PyTorch and the trainer
line [train_model] self-check: importing torch
line @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
stage SelfCheck 100 PyTorch sees the GPU
stage Weights 0 RF-DETR nano COCO checkpoint
stage Weights 50 RF-DETR nano COCO checkpoint 183 of 366 MB
stage Dataset 0 reading the exports
prepared 10 train (10 frames), 4 valid, 5 heldout, classes [\"cat\", \"raccoon\"], heldout dir Some(\"<T>/training/run1/data/heldout\")
stage Dataset 100 10 training, 4 validation, 5 held-out images; classes cat, raccoon
stage Train 0 starting the trainer
line early stopping is off (patience 2 of 2 epochs): every epoch runs; the best one is still exported
stage Train 0 epoch 1/2
progress epoch 1/2 | batch Some(16) | note None | stop 2 0.001 10 3
settings Some(RunSettings { lr: Some(5e-5), lr_encoder: Some(7.5e-5), lr_schedule: Cosine, lr_min_factor: 0.1, select_metric: Map, epochs: 2, patience: 2, patience_doubled: true, batch: 16, workers: Some(0), loaders: Some(2), seed: Some(2), scene: Some(\"night+ir\"), gpu: Some(\"Apple M1 GPU\"), ignore: Some(Background), negative_share: Some(0.15), static_cap: Some(20) })
stage Train 0 epoch 1/9, loss 1.000
progress epoch 1/9, loss 1.000 | batch Some(16) | note None | stop 2 0.001 10 3
line Traceback (most recent call last):
line RuntimeError: MPS backend out of memory (MPS allocated: 11.00 GiB)
line out of GPU memory at batch 16, retrying with batch 8
batch 8
stage Train 0 out of GPU memory at batch 16, retrying with batch 8
line finish requested: the retry stops after its first epoch
stage Train 0 epoch 1/2 (out of GPU memory at batch 16, retrying with batch 8)
progress epoch 1/2 (out of GPU memory at batch 16, retrying with batch 8) | batch Some(8) | note Some(\"out of GPU memory at batch 16, retrying with batch 8\") | stop 2 0.001 10 3
stage Train 0 epoch 1/9, loss 1.000 (out of GPU memory at batch 16, retrying with batch 8)
progress epoch 1/9, loss 1.000 (out of GPU memory at batch 16, retrying with batch 8) | batch Some(8) | note Some(\"out of GPU memory at batch 16, retrying with batch 8\") | stop 2 0.001 10 3
line Traceback (most recent call last):
line RuntimeError: MPS backend out of memory (MPS allocated: 11.00 GiB)
line out of GPU memory at batch 8, retrying with batch 4
batch 4
stage Train 0 out of GPU memory at batch 8, retrying with batch 4
line finish requested: the retry stops after its first epoch
stage Train 0 epoch 1/2 (out of GPU memory at batch 8, retrying with batch 4)
progress epoch 1/2 (out of GPU memory at batch 8, retrying with batch 4) | batch Some(4) | note Some(\"out of GPU memory at batch 8, retrying with batch 4\") | stop 2 0.001 10 3
stage Train 25 epoch 1/2, batch 1/2 (out of GPU memory at batch 8, retrying with batch 4)
progress epoch 1/2, batch 1/2 (out of GPU memory at batch 8, retrying with batch 4) | batch Some(4) | note Some(\"out of GPU memory at batch 8, retrying with batch 4\") | stop 2 0.001 10 3
stage Train 50 epoch 1/2, loss 2.000 (out of GPU memory at batch 8, retrying with batch 4)
progress epoch 1/2, loss 2.000 (out of GPU memory at batch 8, retrying with batch 4) | batch Some(4) | note Some(\"out of GPU memory at batch 8, retrying with batch 4\") | stop 2 0.001 10 3
stage Train 75 epoch 2/2, batch 1/2, loss 2.000 (out of GPU memory at batch 8, retrying with batch 4)
progress epoch 2/2, batch 1/2, loss 2.000 (out of GPU memory at batch 8, retrying with batch 4) | batch Some(4) | note Some(\"out of GPU memory at batch 8, retrying with batch 4\") | stop 2 0.001 10 3
stage Train 100 epoch 2/2, loss 1.000 (out of GPU memory at batch 8, retrying with batch 4)
progress epoch 2/2, loss 1.000 (out of GPU memory at batch 8, retrying with batch 4) | batch Some(4) | note Some(\"out of GPU memory at batch 8, retrying with batch 4\") | stop 2 0.001 10 3
stage Train 100 export: exporting the trained model to ONNX and checking it
progress export: exporting the trained model to ONNX and checking it | batch Some(4) | note Some(\"out of GPU memory at batch 8, retrying with batch 4\") | stop 2 0.001 10 3
resources workers 0 loaders 2 forced true limit None host_oom None None img_s None
stage Install 0 checking the output
installed night-rfdetr-nano-test <T>/models/night-rfdetr-nano-test.onnx [\"cat\", \"raccoon\"] mps
stage Install 100 <T>/models/night-rfdetr-nano-test.onnx
stage Register 0 night-rfdetr-nano-test
registered added night-rfdetr-nano-test
stage Register 100 added night-rfdetr-nano-test
stage Benchmark 0 waiting for the benchmark runner
stage Benchmark 50 comparing with the default model
stage Benchmark 100 done
ok night-rfdetr-nano-test registered \"added night-rfdetr-nano-test\" benchmark Some(BenchSummary { datasets: [], device: None, ms_p50: Some(12.0), ap50: Some(0.61), ap50_95: None, images: 0, threshold: None, person_recall: None, vehicle_recall: None, empty_frames: None, empty_frame_alerts: None, by_scene: {}, in_scope: None, heldout_fingerprint: None, trained_on_share: None, random: None, ap_basis: None, person_free_frames: None, person_false_alerts: None, recall_cells: RecallCells { by_scene: {}, by_camera: {}, by_camera_scene: [] } }) error None
cmd SelfCheck -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --self-check --trainer rfdetr --device mps
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer rfdetr --family rfdetr --size nano --name night-rfdetr-nano-test --weights <T>/tools/train-weights/rfdetr-nano/rf-detr-nano.pth --data <T>/training/run1/data --imgsz 384 --epochs 2 --batch 16 --device mps --workers 0 --seed 2 --aug-dark 0.5 --aug-noise 0.3 --aug-gray 0.5 --out-dir <T>/models/.night-rfdetr-nano-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --lr 0.00005 --lr-encoder 0.000075 --patience 2 --min-epochs 10 --min-delta 0.001 --smooth 3 --anneal 0 --stop-file <T>/training/run1/work/stop-request --multi-scale on
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer rfdetr --family rfdetr --size nano --name night-rfdetr-nano-test --weights <T>/tools/train-weights/rfdetr-nano/rf-detr-nano.pth --data <T>/training/run1/data --imgsz 384 --epochs 2 --batch 8 --device mps --workers 0 --seed 2 --aug-dark 0.5 --aug-noise 0.3 --aug-gray 0.5 --out-dir <T>/models/.night-rfdetr-nano-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --lr 0.00005 --lr-encoder 0.000075 --patience 2 --min-epochs 10 --min-delta 0.001 --smooth 3 --anneal 0 --stop-file <T>/training/run1/work/stop-request --multi-scale on
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer rfdetr --family rfdetr --size nano --name night-rfdetr-nano-test --weights <T>/tools/train-weights/rfdetr-nano/rf-detr-nano.pth --data <T>/training/run1/data --imgsz 384 --epochs 2 --batch 4 --device mps --workers 0 --seed 2 --aug-dark 0.5 --aug-noise 0.3 --aug-gray 0.5 --out-dir <T>/models/.night-rfdetr-nano-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --lr 0.00005 --lr-encoder 0.000075 --patience 2 --min-epochs 10 --min-delta 0.001 --smooth 3 --anneal 0 --stop-file <T>/training/run1/work/stop-request --multi-scale on
log Resolved 48 packages in 3ms
log Downloading torch (75.3MiB)
log  Downloaded torch
log Prepared 48 packages in 40s
log Installed 48 packages in 2s
log [train_model] self-check: importing torch
log @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
log dataset: train / valid images with a short side over 600 px written at 600 px (what the trainer reads at most); held-out frames full size
log dataset: Only 4 training box(es) of 'cat': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 4 training box(es) of 'raccoon': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 5 held-out frame(s): the benchmark of this run is a rough guess, one frame moves it a lot. Review and export frames from more cameras or days.
log dataset: Only 10 IR frame(s) to train on: review more of them.
log dataset: validation on 1 camera day(s) kept out of training: gate/2026-10-01
log # convergence (by the classes with at least 20 validation boxes, every class when none has): 4 validation images (< 50): patience 2 instead of 1; mAP averaged over the last 3 epochs
log # training seed 2 (the frames are the same for every seed)
log # lr: 0.00005, encoder 0.000075 (10 unique training frames < 1000: lower than RF-DETR's default 0.0001 for small datasets); schedule cosine (default), final factor 0.1 (default)
log # early stopping is off (patience 2 of 2 epochs): every epoch runs; the best one is still exported
log # data loader workers: 0 (training.workers)
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log @train {\"event\":\"epoch\",\"epoch\":1,\"epochs\":9,\"loss\":1.0}
log Traceback (most recent call last):
log RuntimeError: MPS backend out of memory (MPS allocated: 11.00 GiB)
log # out of GPU memory at batch 16, retrying with batch 8
log # finish requested: the retry stops after its first epoch
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log @train {\"event\":\"epoch\",\"epoch\":1,\"epochs\":9,\"loss\":1.0}
log Traceback (most recent call last):
log RuntimeError: MPS backend out of memory (MPS allocated: 11.00 GiB)
log # out of GPU memory at batch 8, retrying with batch 4
log # finish requested: the retry stops after its first epoch
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log noise @train {\"event\":\"batch\",\"epoch\":1,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":1,\"epochs\":2,\"loss\":2}
log noise @train {\"event\":\"batch\",\"epoch\":2,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":2,\"epochs\":2,\"loss\":1}
log @train {\"event\":\"phase\",\"phase\":\"export\",\"detail\":\"x\"}
log @train {\"classes\":[\"cat\",\"raccoon\"],\"device\":\"mps\",\"event\":\"done\",\"family\":\"rfdetr\",\"imgsz\":640,\"onnx\":\"<T>/models/.night-rfdetr-nano-test.train-staging/night-rfdetr-nano-test.onnx\",\"seconds\":12.5,\"yaml\":\"y\"}
";
const HOST_OOM_RUN: &str = "\
stage Setup 0 checking the toolchain
stage Setup 0 downloading uv: checking the toolchain
stage Setup 0 downloading uv: uv (model export toolchain)
stage Setup 2 downloading uv: 7.1 of 14.2 MB
stage Setup 5 installing Python 3.11: uv python install 3.11
stage Setup 11 installing Python 3.11: uv venv
stage Setup 16 installing the pinned Python packages: train-macos-arm64.txt
line Resolved 48 packages in 3ms
line Downloading torch (75.3MiB)
stage Setup 16 installing the pinned Python packages: downloading torch (75.3MiB)
line  Downloaded torch
stage Setup 85 installing the pinned Python packages: downloaded torch
line Prepared 48 packages in 40s
stage Setup 90 installing the pinned Python packages: Prepared 48 packages in 40s
line Installed 48 packages in 2s
stage Setup 98 installing the pinned Python packages: Installed 48 packages in 2s
stage Setup 98 installing the pinned Python packages: packages installed
stage Setup 100 checking the Visual C++ runtime: not needed on macos
stage SelfCheck 0 importing PyTorch and the trainer
line [train_model] self-check: importing torch
line @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
stage SelfCheck 100 PyTorch sees the GPU
stage Weights 0 YOLO26 nano
stage Weights 49 YOLO26 nano 2 of 5 MB
stage Dataset 0 reading the exports
prepared 10 train (10 frames), 4 valid, 5 heldout, classes [\"cat\", \"raccoon\"], heldout dir Some(\"<T>/training/run1/data/heldout\")
stage Dataset 100 10 training, 4 validation, 5 held-out images; classes cat, raccoon
stage Train 0 starting the trainer
line early stopping is off (patience 2 of 2 epochs): every epoch runs; the best one is still exported
stage Train 0 epoch 1/2
progress epoch 1/2 | batch Some(8) | note None | stop 2 0.001 10 3
settings Some(RunSettings { lr: None, lr_encoder: None, lr_schedule: Linear, lr_min_factor: 0.01, select_metric: Map, epochs: 2, patience: 2, patience_doubled: true, batch: 8, workers: Some(4), loaders: Some(3), seed: Some(2), scene: Some(\"night+ir\"), gpu: Some(\"Apple M1 GPU\"), ignore: Some(Background), negative_share: Some(0.15), static_cap: Some(20) })
line Traceback (most recent call last):
line   File \"<string>\", line 1, in <module>
line   File \"C:\t/Lib/multiprocessing/spawn.py\", line 122, in spawn_main
line OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading \"C:\t/Lib/site-packages\torch/lib/cufft64_11.dll\" or one of its dependencies.
line OSError: [WinError 1450] Insufficient system resources exist to complete the requested service: 'C:/t/DLLs'
line RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3993600 bytes.
line RuntimeError: can't start new thread
line out of system memory with 4 data loader workers, retrying with 2
stage Train 0 out of system memory with 4 data loader workers, retrying with 2
stage Train 0 epoch 1/2 (out of system memory with 4 data loader workers, retrying with 2)
progress epoch 1/2 (out of system memory with 4 data loader workers, retrying with 2) | batch Some(8) | note Some(\"out of system memory with 4 data loader workers, retrying with 2\") | stop 2 0.001 10 3
settings Some(RunSettings { lr: None, lr_encoder: None, lr_schedule: Linear, lr_min_factor: 0.01, select_metric: Map, epochs: 2, patience: 2, patience_doubled: true, batch: 8, workers: Some(2), loaders: Some(3), seed: Some(2), scene: Some(\"night+ir\"), gpu: Some(\"Apple M1 GPU\"), ignore: Some(Background), negative_share: Some(0.15), static_cap: Some(20) })
line Traceback (most recent call last):
line   File \"<string>\", line 1, in <module>
line   File \"C:\t/Lib/multiprocessing/spawn.py\", line 122, in spawn_main
line OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading \"C:\t/Lib/site-packages\torch/lib/cufft64_11.dll\" or one of its dependencies.
line OSError: [WinError 1450] Insufficient system resources exist to complete the requested service: 'C:/t/DLLs'
line RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3993600 bytes.
line RuntimeError: can't start new thread
line out of system memory with 2 data loader workers, retrying with 1
stage Train 0 out of system memory with 2 data loader workers, retrying with 1
stage Train 0 epoch 1/2 (out of system memory with 2 data loader workers, retrying with 1)
progress epoch 1/2 (out of system memory with 2 data loader workers, retrying with 1) | batch Some(8) | note Some(\"out of system memory with 2 data loader workers, retrying with 1\") | stop 2 0.001 10 3
settings Some(RunSettings { lr: None, lr_encoder: None, lr_schedule: Linear, lr_min_factor: 0.01, select_metric: Map, epochs: 2, patience: 2, patience_doubled: true, batch: 8, workers: Some(1), loaders: Some(3), seed: Some(2), scene: Some(\"night+ir\"), gpu: Some(\"Apple M1 GPU\"), ignore: Some(Background), negative_share: Some(0.15), static_cap: Some(20) })
stage Train 25 epoch 1/2, batch 1/2 (out of system memory with 2 data loader workers, retrying with 1)
progress epoch 1/2, batch 1/2 (out of system memory with 2 data loader workers, retrying with 1) | batch Some(8) | note Some(\"out of system memory with 2 data loader workers, retrying with 1\") | stop 2 0.001 10 3
stage Train 50 epoch 1/2, loss 2.000 (out of system memory with 2 data loader workers, retrying with 1)
progress epoch 1/2, loss 2.000 (out of system memory with 2 data loader workers, retrying with 1) | batch Some(8) | note Some(\"out of system memory with 2 data loader workers, retrying with 1\") | stop 2 0.001 10 3
stage Train 75 epoch 2/2, batch 1/2, loss 2.000 (out of system memory with 2 data loader workers, retrying with 1)
progress epoch 2/2, batch 1/2, loss 2.000 (out of system memory with 2 data loader workers, retrying with 1) | batch Some(8) | note Some(\"out of system memory with 2 data loader workers, retrying with 1\") | stop 2 0.001 10 3
stage Train 100 epoch 2/2, loss 1.000 (out of system memory with 2 data loader workers, retrying with 1)
progress epoch 2/2, loss 1.000 (out of system memory with 2 data loader workers, retrying with 1) | batch Some(8) | note Some(\"out of system memory with 2 data loader workers, retrying with 1\") | stop 2 0.001 10 3
stage Train 100 export: exporting the trained model to ONNX and checking it
progress export: exporting the trained model to ONNX and checking it | batch Some(8) | note Some(\"out of system memory with 2 data loader workers, retrying with 1\") | stop 2 0.001 10 3
resources workers 1 loaders 3 forced true limit None host_oom Some(2) None img_s None
stage Install 0 checking the output
installed night-yolo26n-test <T>/models/night-yolo26n-test.onnx [\"cat\", \"raccoon\"] mps
stage Install 100 <T>/models/night-yolo26n-test.onnx
stage Register 0 night-yolo26n-test
registered night-yolo26n-test installed (not added to the config)
stage Register 100 night-yolo26n-test installed (not added to the config)
stage Benchmark 0 waiting for the benchmark runner
stage Benchmark 100 done
ok night-yolo26n-test registered \"night-yolo26n-test installed (not added to the config)\" benchmark None error None
cmd SelfCheck -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --self-check --trainer ultralytics --device mps
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer ultralytics --family yolo26 --size n --name night-yolo26n-test --weights <T>/tools/export-weights/yolo26n/yolo26n.pt --data <T>/training/run1/data --imgsz 640 --epochs 2 --batch 8 --device mps --workers 4 --seed 2 --aug-dark 0.5 --aug-noise 0.3 --aug-gray 0.5 --out-dir <T>/models/.night-yolo26n-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --patience 2 --min-epochs 10 --min-delta 0.001 --smooth 3 --anneal 0 --stop-file <T>/training/run1/work/stop-request
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer ultralytics --family yolo26 --size n --name night-yolo26n-test --weights <T>/tools/export-weights/yolo26n/yolo26n.pt --data <T>/training/run1/data --imgsz 640 --epochs 2 --batch 8 --device mps --workers 2 --seed 2 --aug-dark 0.5 --aug-noise 0.3 --aug-gray 0.5 --out-dir <T>/models/.night-yolo26n-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --patience 2 --min-epochs 10 --min-delta 0.001 --smooth 3 --anneal 0 --stop-file <T>/training/run1/work/stop-request
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer ultralytics --family yolo26 --size n --name night-yolo26n-test --weights <T>/tools/export-weights/yolo26n/yolo26n.pt --data <T>/training/run1/data --imgsz 640 --epochs 2 --batch 8 --device mps --workers 1 --seed 2 --aug-dark 0.5 --aug-noise 0.3 --aug-gray 0.5 --out-dir <T>/models/.night-yolo26n-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --patience 2 --min-epochs 10 --min-delta 0.001 --smooth 3 --anneal 0 --stop-file <T>/training/run1/work/stop-request
log Resolved 48 packages in 3ms
log Downloading torch (75.3MiB)
log  Downloaded torch
log Prepared 48 packages in 40s
log Installed 48 packages in 2s
log [train_model] self-check: importing torch
log @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
log dataset: train / valid images with a short side over 960 px written at 960 px (what the trainer reads at most); held-out frames full size
log dataset: Only 4 training box(es) of 'cat': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 4 training box(es) of 'raccoon': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 5 held-out frame(s): the benchmark of this run is a rough guess, one frame moves it a lot. Review and export frames from more cameras or days.
log dataset: Only 10 IR frame(s) to train on: review more of them.
log dataset: validation on 1 camera day(s) kept out of training: gate/2026-10-01
log # convergence (by the classes with at least 20 validation boxes, every class when none has): 4 validation images (< 50): patience 2 instead of 1; mAP averaged over the last 3 epochs
log # training seed 2 (the frames are the same for every seed)
log # lr: 10 unique training frames: the trainer's default; schedule linear (default), final factor 0.01 (default)
log # early stopping is off (patience 2 of 2 epochs): every epoch runs; the best one is still exported
log # data loader workers: 4 (training.workers)
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log Traceback (most recent call last):
log   File \"<string>\", line 1, in <module>
log   File \"C:\t/Lib/multiprocessing/spawn.py\", line 122, in spawn_main
log OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading \"C:\t/Lib/site-packages\torch/lib/cufft64_11.dll\" or one of its dependencies.
log OSError: [WinError 1450] Insufficient system resources exist to complete the requested service: 'C:/t/DLLs'
log RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3993600 bytes.
log RuntimeError: can't start new thread
log # out of system memory with 4 data loader workers, retrying with 2
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log Traceback (most recent call last):
log   File \"<string>\", line 1, in <module>
log   File \"C:\t/Lib/multiprocessing/spawn.py\", line 122, in spawn_main
log OSError: [WinError 1455] The paging file is too small for this operation to complete. Error loading \"C:\t/Lib/site-packages\torch/lib/cufft64_11.dll\" or one of its dependencies.
log OSError: [WinError 1450] Insufficient system resources exist to complete the requested service: 'C:/t/DLLs'
log RuntimeError: [enforce fail at alloc_cpu.cpp:117] data. DefaultCPUAllocator: not enough memory: you tried to allocate 3993600 bytes.
log RuntimeError: can't start new thread
log # out of system memory with 2 data loader workers, retrying with 1
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log noise @train {\"event\":\"batch\",\"epoch\":1,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":1,\"epochs\":2,\"loss\":2}
log noise @train {\"event\":\"batch\",\"epoch\":2,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":2,\"epochs\":2,\"loss\":1}
log @train {\"event\":\"phase\",\"phase\":\"export\",\"detail\":\"x\"}
log @train {\"classes\":[\"cat\",\"raccoon\"],\"device\":\"mps\",\"event\":\"done\",\"family\":\"yolo26\",\"imgsz\":640,\"onnx\":\"<T>/models/.night-yolo26n-test.train-staging/night-yolo26n-test.onnx\",\"seconds\":12.5,\"yaml\":\"y\"}
";
const FAILED_EXPORT: &str = "\
stage Setup 0 checking the toolchain
stage Setup 0 downloading uv: checking the toolchain
stage Setup 0 downloading uv: uv (model export toolchain)
stage Setup 2 downloading uv: 7.1 of 14.2 MB
stage Setup 5 installing Python 3.11: uv python install 3.11
stage Setup 11 installing Python 3.11: uv venv
stage Setup 16 installing the pinned Python packages: train-macos-arm64.txt
line Resolved 48 packages in 3ms
line Downloading torch (75.3MiB)
stage Setup 16 installing the pinned Python packages: downloading torch (75.3MiB)
line  Downloaded torch
stage Setup 85 installing the pinned Python packages: downloaded torch
line Prepared 48 packages in 40s
stage Setup 90 installing the pinned Python packages: Prepared 48 packages in 40s
line Installed 48 packages in 2s
stage Setup 98 installing the pinned Python packages: Installed 48 packages in 2s
stage Setup 98 installing the pinned Python packages: packages installed
stage Setup 100 checking the Visual C++ runtime: not needed on macos
stage SelfCheck 0 importing PyTorch and the trainer
line [train_model] self-check: importing torch
line @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
stage SelfCheck 100 PyTorch sees the GPU
stage Weights 0 YOLO26 nano
stage Weights 49 YOLO26 nano 2 of 5 MB
stage Dataset 0 reading the exports
prepared 10 train (10 frames), 4 valid, 5 heldout, classes [\"cat\", \"raccoon\"], heldout dir Some(\"<T>/training/run1/data/heldout\")
stage Dataset 100 10 training, 4 validation, 5 held-out images; classes cat, raccoon
stage Train 0 starting the trainer
line early stopping is off (patience 2 of 2 epochs): every epoch runs; the best one is still exported
stage Train 0 epoch 1/2
progress epoch 1/2 | batch Some(8) | note None | stop 2 0.001 10 3
settings Some(RunSettings { lr: None, lr_encoder: None, lr_schedule: Linear, lr_min_factor: 0.01, select_metric: Map, epochs: 2, patience: 2, patience_doubled: true, batch: 8, workers: Some(0), loaders: Some(3), seed: Some(2), scene: Some(\"night+ir\"), gpu: Some(\"Apple M1 GPU\"), ignore: Some(Background), negative_share: Some(0.15), static_cap: Some(20) })
stage Train 25 epoch 1/2, batch 1/2
progress epoch 1/2, batch 1/2 | batch Some(8) | note None | stop 2 0.001 10 3
stage Train 50 epoch 1/2, loss 2.000
progress epoch 1/2, loss 2.000 | batch Some(8) | note None | stop 2 0.001 10 3
stage Train 75 epoch 2/2, batch 1/2, loss 2.000
progress epoch 2/2, batch 1/2, loss 2.000 | batch Some(8) | note None | stop 2 0.001 10 3
stage Train 100 epoch 2/2, loss 1.000
progress epoch 2/2, loss 1.000 | batch Some(8) | note None | stop 2 0.001 10 3
stage Train 100 export: exporting the trained model to ONNX and checking it
progress export: exporting the trained model to ONNX and checking it | batch Some(8) | note None | stop 2 0.001 10 3
line RuntimeError: ONNX Runtime and PyTorch disagree
resources workers 0 loaders 3 forced true limit None host_oom None None img_s None
err Some(Train) python failed (exit status: 1); last output: RuntimeError: ONNX Runtime and PyTorch disagree. Its best checkpoint is kept (2 kB in <T>/training/run1/work): Retry export on the run's row (train retry-export <ID>) makes the model without training again cancelled false
cmd SelfCheck -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --self-check --trainer ultralytics --device mps
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer ultralytics --family yolo26 --size n --name night-yolo26n-test --weights <T>/tools/export-weights/yolo26n/yolo26n.pt --data <T>/training/run1/data --imgsz 640 --epochs 2 --batch 8 --device mps --workers 0 --seed 2 --aug-dark 0.5 --aug-noise 0.3 --aug-gray 0.5 --out-dir <T>/models/.night-yolo26n-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --patience 2 --min-epochs 10 --min-delta 0.001 --smooth 3 --anneal 0 --stop-file <T>/training/run1/work/stop-request
log Resolved 48 packages in 3ms
log Downloading torch (75.3MiB)
log  Downloaded torch
log Prepared 48 packages in 40s
log Installed 48 packages in 2s
log [train_model] self-check: importing torch
log @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
log dataset: train / valid images with a short side over 960 px written at 960 px (what the trainer reads at most); held-out frames full size
log dataset: Only 4 training box(es) of 'cat': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 4 training box(es) of 'raccoon': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 5 held-out frame(s): the benchmark of this run is a rough guess, one frame moves it a lot. Review and export frames from more cameras or days.
log dataset: Only 10 IR frame(s) to train on: review more of them.
log dataset: validation on 1 camera day(s) kept out of training: gate/2026-10-01
log # convergence (by the classes with at least 20 validation boxes, every class when none has): 4 validation images (< 50): patience 2 instead of 1; mAP averaged over the last 3 epochs
log # training seed 2 (the frames are the same for every seed)
log # lr: 10 unique training frames: the trainer's default; schedule linear (default), final factor 0.01 (default)
log # early stopping is off (patience 2 of 2 epochs): every epoch runs; the best one is still exported
log # data loader workers: 0 (training.workers)
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log noise @train {\"event\":\"batch\",\"epoch\":1,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":1,\"epochs\":2,\"loss\":2}
log noise @train {\"event\":\"batch\",\"epoch\":2,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":2,\"epochs\":2,\"loss\":1}
log @train {\"event\":\"phase\",\"phase\":\"export\",\"detail\":\"x\"}
log RuntimeError: ONNX Runtime and PyTorch disagree
";
const EXPORT_RETRY: &str = "\
stage Setup 0 checking the toolchain
stage Setup 0 downloading uv: checking the toolchain
stage Setup 5 downloading uv: uv is installed
stage Setup 16 installing Python 3.11: reusing the Python environment
stage Setup 98 installing the pinned Python packages: packages already installed
stage Setup 100 checking the Visual C++ runtime: not needed on macos
stage SelfCheck 0 importing PyTorch and the trainer
line [train_model] self-check: importing torch
line @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
stage SelfCheck 100 PyTorch sees the GPU
stage Train 0 exporting the best checkpoint
stage Train 50 exporting best.pt
stage Install 0 checking the output
installed night-yolo26n-test <T>/models/night-yolo26n-test.onnx [\"cat\", \"raccoon\"] NVIDIA RTX
stage Install 100 <T>/models/night-yolo26n-test.onnx
stage Register 0 night-yolo26n-test
registered added night-yolo26n-test
stage Register 100 added night-yolo26n-test
stage Benchmark 0 waiting for the benchmark runner
stage Benchmark 50 comparing with the default model
stage Benchmark 100 done
ok night-yolo26n-test registered \"added night-yolo26n-test\" benchmark Some(BenchSummary { datasets: [], device: None, ms_p50: Some(12.0), ap50: Some(0.61), ap50_95: None, images: 0, threshold: None, person_recall: None, vehicle_recall: None, empty_frames: None, empty_frame_alerts: None, by_scene: {}, in_scope: None, heldout_fingerprint: None, trained_on_share: None, random: None, ap_basis: None, person_free_frames: None, person_false_alerts: None, recall_cells: RecallCells { by_scene: {}, by_camera: {}, by_camera_scene: [] } }) error None
cmd SelfCheck -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --self-check --trainer ultralytics --device mps
cmd Export -X faulthandler -X utf8 -u -I <T>/tools/train-work/train_model.py --trainer ultralytics --family yolo26 --size n --name night-yolo26n-test --data <T>/training/run1/data --imgsz 640 --out-dir <T>/models/.night-yolo26n-test.train-staging --work <T>/training/run1/work --export-script <T>/tools/train-work/export_model.py --export-only <T>/training/run1/work/run/weights/best.pt
log Resolved 48 packages in 3ms
log Downloading torch (75.3MiB)
log  Downloaded torch
log Prepared 48 packages in 40s
log Installed 48 packages in 2s
log [train_model] self-check: importing torch
log @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
log dataset: train / valid images with a short side over 960 px written at 960 px (what the trainer reads at most); held-out frames full size
log dataset: Only 4 training box(es) of 'cat': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 4 training box(es) of 'raccoon': the model will find it poorly. Review more frames with it, or leave the class out.
log dataset: Only 5 held-out frame(s): the benchmark of this run is a rough guess, one frame moves it a lot. Review and export frames from more cameras or days.
log dataset: Only 10 IR frame(s) to train on: review more of them.
log dataset: validation on 1 camera day(s) kept out of training: gate/2026-10-01
log # convergence (by the classes with at least 20 validation boxes, every class when none has): 4 validation images (< 50): patience 2 instead of 1; mAP averaged over the last 3 epochs
log # training seed 2 (the frames are the same for every seed)
log # lr: 10 unique training frames: the trainer's default; schedule linear (default), final factor 0.01 (default)
log # early stopping is off (patience 2 of 2 epochs): every epoch runs; the best one is still exported
log # data loader workers: 0 (training.workers)
log @train {\"event\":\"start\",\"device\":\"mps\",\"train_images\":20,\"valid_images\":2,\"epochs\":2}
log noise @train {\"event\":\"batch\",\"epoch\":1,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":1,\"epochs\":2,\"loss\":2}
log noise @train {\"event\":\"batch\",\"epoch\":2,\"epochs\":2,\"batch\":1,\"batches\":2}
log @train {\"event\":\"epoch\",\"epoch\":2,\"epochs\":2,\"loss\":1}
log @train {\"event\":\"phase\",\"phase\":\"export\",\"detail\":\"x\"}
log RuntimeError: ONNX Runtime and PyTorch disagree
log [train_model] self-check: importing torch
log @train {\"event\":\"self_check\",\"cuda\":false,\"mps\":true,\"cuda_ok\":true,\"torch\":\"2.14.0\"}
log @train {\"event\":\"phase\",\"phase\":\"export\",\"detail\":\"exporting best.pt\"}
log @train {\"classes\":[\"cat\",\"raccoon\"],\"device\":\"mps\",\"event\":\"done\",\"family\":\"yolo26\",\"imgsz\":640,\"onnx\":\"<T>/models/.night-yolo26n-test.train-staging/night-yolo26n-test.onnx\",\"seconds\":12.5,\"yaml\":\"y\"}
";
