# Production review: Blue Onyx Prism

*Phase 0 of the production-hardening review, 2026-10-12. Read-only: this document records what was
found before anything was changed. Later phases append to the status table at the end.*

## How this review was done

- Every tracked file (`git ls-files`, 157 files) was opened end to end by one reviewer. The tree was split
  by directory among eight read-only subagents (four Fable, four Opus); the lead merged the ledgers,
  re-ran the global sweeps (panic sites, `unsafe` sites, process spawns, listeners, HTTP clients) and
  spot-checked one or more findings from every report against the code (CO-01/02/03, PL-01/03,
  SV-01/08, BE-01/02 all confirmed as written).
- Rubric per Rust file: inputs, failure paths, resources, concurrency, platform, correctness,
  API/config stability, tests, readability. Non-Rust files used a per-type rubric (templates and JS: escaping, CSP, admin posts; scripts: strict
  mode, quoting, pinning; workflows: SHA pins, permissions, `--locked`; deploy: least privilege; docs:
  every command and key exists).
- Baseline on this machine (macOS arm64, no NVIDIA, no Windows), commit `e394733`:
  `cargo fmt --all -- --check` clean, `cargo clippy --locked --all-targets -- -D warnings` clean,
  `cargo test --locked`: 469 tests passed, 0 failed, 18 ignored (need `models/` + runtime libs).
  `cargo deny check` (bans, licenses, sources) passes offline; the advisories check needs network.
- Global sweep: 23 non-test `unwrap`/`expect`/`panic!`/`unreachable!` sites outside `src/server/test_util.rs`
  (all classified below); 47 `unsafe` lines, all in `src/backend/{detect,device,libs,ort}.rs`; one
  `Command::new` that runs downloaded code (`src/resources/export.rs:587`) plus `sysctl`/`powershell` probes
  in `src/system_info.rs`; one HTTP listener (`src/server.rs:310`, plus best-effort `[::1]`); two outbound
  HTTP clients (`src/resources/manager.rs:753`, `src/update.rs:34`).

## Coverage ledger

205 rows, zero *not reviewed*. Files over 1500 lines have one row per section or line range.
Binary fixtures (`.jpg`, `.png`, `.ico`, `.ttf`) are listed as reviewed by inspection of what reads them.

| path | lines | category | reviewer | verdict |
|---|---|---|---|---|
| .claude/agents/engineer.md | 20 | docs | non-Rust+tests (Opus subagent) | clean |
| .claude/agents/porter.md | 19 | docs | non-Rust+tests (Opus subagent) | findings: NR-23 |
| .dockerignore | 7 | build | non-Rust+tests (Opus subagent) | clean |
| .gitattributes | 6 | build | non-Rust+tests (Opus subagent) | clean |
| .github/dependabot.yml | 16 | workflow | non-Rust+tests (Opus subagent) | clean |
| .github/workflows/check-urls.yml | 25 | workflow | non-Rust+tests (Opus subagent) | findings: NR-20 |
| .github/workflows/ci.yml | 115 | workflow | non-Rust+tests (Opus subagent) | findings: NR-05, NR-14, NR-20 |
| .github/workflows/release.yml | 149 | workflow | non-Rust+tests (Opus subagent) | findings: NR-05, NR-07, NR-17 |
| .gitignore | 34 | build | non-Rust+tests (Opus subagent) | clean |
| CLAUDE.md | 126 | docs | non-Rust+tests (Opus subagent) | findings: NR-08, NR-23 |
| Cargo.lock | 3486 | build | lead | clean: lockfile v4, 361 packages, every package but the crate itself has a crates.io checksum, no git or path sources; `--locked` builds pass; duplicate versions limited to base64 0.22/0.23, libloading 0.8/0.9, miniz_oxide 0.8/0.9, syn 2/3 (transitive, see NR-22) |
| Cargo.toml | 103 | build | non-Rust+tests (Opus subagent) | findings: NR-22 |
| Dockerfile | 38 | deploy | non-Rust+tests (Opus subagent) | findings: NR-18, NR-19 |
| LICENSE | 49 | docs | non-Rust+tests (Opus subagent) | findings: NR-17 |
| README.md | 728 | docs | non-Rust+tests (Opus subagent) | findings: NR-09, NR-10, NR-11 |
| assets/bench/bmd45-cctv.json | 2961 | asset | benchmark-metrics (Opus subagent) | clean (skimmed) |
| assets/bench/coco-cctv.json | 9711 | asset | benchmark-metrics (Opus subagent) | clean (skimmed; see BQ-06 for crowd regions) |
| assets/bench/exdark-night.json | 7185 | asset | benchmark-metrics (Opus subagent) | clean (skimmed) |
| assets/coco_classes.yaml | 81 | asset | core/cli/config (Opus subagent) | clean (see the naming note in the rubric) |
| assets/favicon.ico | binary (228 B) | asset | non-Rust+tests (Opus subagent) | clean |
| assets/live.js | 408 | asset | server+UI (Fable subagent) | findings: SV-11 |
| assets/roboto-mono-stripped.ttf | binary (12 KB) | asset | non-Rust+tests (Opus subagent) | findings: NR-17 (Apache-2.0 notice not shipped) |
| assets/style.css | 438 | asset | server+UI (Fable subagent) | clean |
| deny.toml | 44 | build | non-Rust+tests (Opus subagent) | findings: NR-25 |
| deploy/blue-onyx-prism.service | 19 | deploy | non-Rust+tests (Opus subagent) | findings: NR-04 |
| deploy/com.blueonyx.prism.plist | 22 | deploy | non-Rust+tests (Opus subagent) | findings: NR-03, NR-06 |
| deploy/docker-compose.yml | 30 | deploy | non-Rust+tests (Opus subagent) | findings: NR-01, NR-09, NR-19 |
| deploy/docker-config.json | 14 | deploy | non-Rust+tests (Opus subagent) | findings: NR-01 |
| docs/BENCH_IMAGES.md | 138 | docs | non-Rust+tests (Opus subagent) | clean (counts 44/160/180, 142.4/6.7/31.5 MB and 363 scored BMD objects match the manifests) |
| docs/PLAN.md | 411 | docs | non-Rust+tests (Opus subagent) | clean (marked historical in line 1; stale bits such as `export_yolo26.py` are acceptable under that banner) |
| docs/RELEASE_CHECKLIST.md | 33 | docs | non-Rust+tests (Opus subagent) | findings: NR-12, NR-13 |
| docs/REVIEW_PLAN.md | 425 | docs | non-Rust+tests (Opus subagent) | findings: NR-12 (line 369 repeats the GTX 1070 Ti step) |
| rust-toolchain.toml | 2 | build | non-Rust+tests (Opus subagent) | findings: NR-25 |
| scripts/build_bench_sets.py | 771 | script | non-Rust+tests (Opus subagent) | findings: NR-26 |
| scripts/ci/smoke.sh | 72 | script | non-Rust+tests (Opus subagent) | findings: NR-15 |
| scripts/convert_onnx_to_ir.py | 90 | script | non-Rust+tests (Opus subagent) | clean |
| scripts/export_model.py | 728 | script | non-Rust+tests (Opus subagent) | findings: NR-08 |
| scripts/install_launchd.sh | 29 | script | non-Rust+tests (Opus subagent) | findings: NR-03, NR-21 |
| scripts/install_service.ps1 | 95 | script | non-Rust+tests (Opus subagent) | findings: NR-02, NR-16, NR-21 |
| scripts/install_systemd.sh | 29 | script | non-Rust+tests (Opus subagent) | findings: NR-04, NR-21 |
| scripts/model-export/gen_locks.sh | 41 | script | non-Rust+tests (Opus subagent) | clean |
| scripts/model-export/locks/macos-arm64.txt | 1325 | build | non-Rust+tests (Opus subagent) | clean (format checked by script: 63 pinned requirements, every one `==` and `--hash=sha256`; not read line by line) |
| scripts/model-export/locks/torch-cpu-index.txt | 1334 | build | non-Rust+tests (Opus subagent) | clean (same check: 63/63 pinned and hashed; same package set as macOS) |
| scripts/requirements-bench.txt | 4 | script | non-Rust+tests (Opus subagent) | findings: NR-26 |
| scripts/requirements-export.txt | 15 | script | non-Rust+tests (Opus subagent) | clean (ranges on purpose: by-hand env) |
| scripts/requirements-model-export.txt | 26 | script | non-Rust+tests (Opus subagent) | clean |
| scripts/uninstall_service.ps1 | 38 | script | non-Rust+tests (Opus subagent) | findings: NR-21 |
| src/api.rs | 168 | core | pipeline (Fable subagent) | clean |
| src/backend/detect.rs 1-430 (types, `detect`, `attach_cuda`, heuristics) | 430 | backend | backend (Fable subagent) | clean |
| src/backend/detect.rs 430-600 (parsers, `linux` module) | 170 | backend | backend (Fable subagent) | clean |
| src/backend/detect.rs 600-802 (`dxgi`, `cuda_driver` unsafe FFI) | 202 | backend | backend (Fable subagent) | findings: BE-01 (minor) |
| src/backend/detect.rs 803-1120 (tests) | 318 | test | backend (Fable subagent) | clean |
| src/backend/device.rs 1-520 (properties, introspection, reshape, `load_model`) | 520 | backend | backend (Fable subagent) | clean |
| src/backend/device.rs 1050-1194 (tests) | 144 | test | backend (Fable subagent) | clean |
| src/backend/device.rs 520-1050 (callback FFI, slots, tensor IO, submit/collect) | 530 | backend | backend (Fable subagent) | findings: BE-11, BE-14 |
| src/backend/libs.rs 1-420 (OpenVINO env, preload, variadic set_property) | 420 | backend | backend (Fable subagent) | findings: BE-08, BE-11, BE-15 |
| src/backend/libs.rs 420-900 (ORT lookup, CUDA libs preload, probes) | 480 | backend | backend (Fable subagent) | findings: BE-01, BE-02 |
| src/backend/libs.rs 900-1214 (tests) | 314 | test | backend (Fable subagent) | clean |
| src/backend/mod.rs | 781 | backend | backend (Fable subagent) | findings: BE-09 |
| src/backend/onnx_rewrite.rs 1-250 (wire format reader/writer) | 250 | backend | backend (Fable subagent) | findings: BE-06 |
| src/backend/onnx_rewrite.rs 1000-1290 (`rewrite`, `assemble`, `reencode`, `rewrite_cached`) | 290 | backend | backend (Fable subagent) | findings: BE-04, BE-06 |
| src/backend/onnx_rewrite.rs 1290-2191 (tests) | 900 | test | backend (Fable subagent) | clean |
| src/backend/onnx_rewrite.rs 250-560 (ONNX messages, `Graph::parse`) | 310 | backend | backend (Fable subagent) | findings: BE-06 |
| src/backend/onnx_rewrite.rs 560-1000 (builder, emitters, `split_sizes`, `emit_flatten`) | 440 | backend | backend (Fable subagent) | findings: BE-05 |
| src/backend/ort.rs 1-450 (library load, NVIDIA dep probes, `OrtRuntime::new`) | 450 | backend | backend (Fable subagent) | findings: BE-01, BE-02 |
| src/backend/ort.rs 1400-1630 (tests) | 230 | test | backend (Fable subagent) | clean |
| src/backend/ort.rs 450-900 (`compile`, `full_name`, CoreML cache, `build_session`, `unsupported_on`) | 450 | backend | backend (Fable subagent) | findings: BE-03, BE-04, BE-07 |
| src/backend/ort.rs 900-1400 (ports, overrides, session threads, tensor IO, Drop) | 500 | backend | backend (Fable subagent) | findings: BE-10 |
| src/backend/plan.rs | 493 | backend | backend (Fable subagent) | findings: BE-13 |
| src/backend/select.rs 1-400 (probe types, `Selection`) | 400 | backend | backend (Fable subagent) | findings: BE-12 |
| src/backend/select.rs 400-830 (`select`, ranking, `cuda_ordinals`, formatting) | 430 | backend | backend (Fable subagent) | clean |
| src/backend/select.rs 830-1050 (tests) | 220 | test | backend (Fable subagent) | clean |
| src/backend/spec.rs | 477 | backend | backend (Fable subagent) | findings: BE-07 |
| src/benchmark/cli.rs (BenchArgs, arg parsers, ModelReport/Report) | 1-247 | cli | benchmark-core (Opus subagent) | findings: BM-02, BM-04 |
| src/benchmark/cli.rs (impl RunCtx: pseudo GT, run_jobs, sweep_job, run_job, finish_sweep, save_results, apply_devices, write_report) | 784-1097 | cli | benchmark-core (Opus subagent) | findings: BM-01 |
| src/benchmark/cli.rs (main, load_config, jobs, dataset_refs, load_sets, list_datasets, run, check_args, timing_settings, cache_dir_arg, stored_threshold_search, open_runtimes, print_header, cli_progress, impl Report) | 248-783 | cli | benchmark-core (Opus subagent) | findings: BM-02, BM-04 |
| src/benchmark/cli.rs (printing helpers, model_report, agreement_text, device tables) | 1361-1955 | cli | benchmark-core (Opus subagent) | findings: BM-07 |
| src/benchmark/cli.rs (tests) | 1956-2307 | test | benchmark-core (Opus subagent) | clean |
| src/benchmark/cli.rs (threshold_search_cli, search_request, print_search_*, apply_thresholds) | 1099-1360 | cli | benchmark-core (Opus subagent) | clean |
| src/benchmark/current.rs | 368 | benchmark | benchmark-core (Opus subagent) | clean |
| src/benchmark/export.rs | 1008 | benchmark | benchmark-core (Opus subagent) | findings: BM-08 |
| src/benchmark/grade.rs | 421 | benchmark | benchmark-metrics (Opus subagent) | findings: BQ-12 |
| src/benchmark/images.rs | 1151 | benchmark | benchmark-metrics (Opus subagent) | findings: BQ-01, BQ-03, BQ-05, BQ-08, BQ-09, BQ-10 |
| src/benchmark/metrics.rs | 1343 | benchmark | benchmark-metrics (Opus subagent) | findings: BQ-06, BQ-11 |
| src/benchmark/mod.rs (Pipeline, Timings, image_run) | 1060-1204 | benchmark | benchmark-core (Opus subagent) | findings: BM-10 |
| src/benchmark/mod.rs (find_model, job_for_*, configured_device, pick_reference, Phase/Progress, Bench struct, helpers) | 500-741 | benchmark | benchmark-core (Opus subagent) | clean |
| src/benchmark/mod.rs (impl Bench: run / run_exact / run_candidates / time / warm_up / timed_runs) | 742-1058 | benchmark | benchmark-core (Opus subagent) | findings: BM-04 |
| src/benchmark/mod.rs (pseudo_ground_truth, REFERENCE_DEVICES, pseudo_gt_device, SweepOptions, sweep) | 1632-1838 | benchmark | benchmark-core (Opus subagent) | findings: BM-10, BM-11 |
| src/benchmark/mod.rs (score, score_image, all_classes_ap, Scored, CurveMaker) | 1206-1630 | benchmark | benchmark-core (Opus subagent) | clean |
| src/benchmark/mod.rs (tests) | 1839-2636 | test | benchmark-core (Opus subagent) | clean |
| src/benchmark/mod.rs (types, Stats/percentile, Detection, RunResult, compare, Job helpers) | 1-500 | benchmark | benchmark-core (Opus subagent) | findings: BM-04, BM-07, BM-09 |
| src/benchmark/report.rs (ACCURACY_TIE, Ranked, speed_order, recommend) | 211-403 | benchmark | benchmark-core (Opus subagent) | clean |
| src/benchmark/report.rs (Agreement, verdicts, DeviceResult) | 1-210 | benchmark | benchmark-core (Opus subagent) | clean |
| src/benchmark/report.rs (BenchmarkResults + impl: ranking, regrade, load/save/merge, auto_order) | 729-988 | benchmark | benchmark-core (Opus subagent) | findings: BM-01, BM-03 |
| src/benchmark/report.rs (HardwareSummary, RuntimeVersions, ImageInfo/ImageSetInfo, RankRow) | 579-728 | benchmark | benchmark-core (Opus subagent) | findings: BM-03 |
| src/benchmark/report.rs (ModelResult + impl) | 404-578 | benchmark | benchmark-core (Opus subagent) | clean |
| src/benchmark/report.rs (tests) | 989-1912 | test | benchmark-core (Opus subagent) | clean |
| src/benchmark/roc.rs | 519 | benchmark | benchmark-metrics (Opus subagent) | findings: BQ-11 |
| src/benchmark/search.rs | 1281 | benchmark | benchmark-metrics (Opus subagent) | findings: BQ-02, BQ-04, BQ-13 |
| src/benchmark/service.rs | 1173 | benchmark | benchmark-core (Opus subagent) | findings: BM-01, BM-05, BM-06, BM-11 |
| src/benchmark/threshold.rs:1-216 (Objective, parse/print) | 216 | benchmark | benchmark-metrics (Opus subagent) | clean |
| src/benchmark/threshold.rs:217-523 (Pick, segments, plateau, exact_pick, grid pick, fallback) | 307 | benchmark | benchmark-metrics (Opus subagent) | findings: BQ-07 |
| src/benchmark/threshold.rs:524-909 (Curve, ThresholdSweep, GroupPick, ThresholdAdvice, advise, table_points) | 386 | benchmark | benchmark-metrics (Opus subagent) | findings: BQ-07 |
| src/benchmark/threshold.rs:910-1523 (test_advice + tests) | 614 | test | benchmark-metrics (Opus subagent) | clean |
| src/bin/blue_onyx_prism.rs | 392 | cli | core/cli/config (Opus subagent) | findings: CO-10, CO-11, CO-12 |
| src/bin/blue_onyx_prism_benchmark.rs | 29 | cli | core/cli/config (Opus subagent) | clean |
| src/bin/blue_onyx_prism_service.rs | 341 | core | core/cli/config (Opus subagent) | findings: CO-05, CO-12, CO-15 (not compiled here, Windows only) |
| src/bin/test_blue_onyx_prism.rs | 237 | test | core/cli/config (Opus subagent) | findings: CO-14 |
| src/cli.rs | 907 | cli | core/cli/config (Opus subagent) | findings: CO-01, CO-02, CO-04, CO-05, CO-06, CO-07, CO-11 |
| src/config.rs | 969 | core | core/cli/config (Opus subagent) | findings: CO-02, CO-03, CO-04, CO-09, CO-16, CO-17 |
| src/config_form.rs | 690 | core | core/cli/config (Opus subagent) | findings: CO-03, CO-04, CO-16 |
| src/config_merge.rs | 854 | core | core/cli/config (Opus subagent) | findings: CO-09 |
| src/config_store.rs | 456 | core | core/cli/config (Opus subagent) | findings: CO-02, CO-04, CO-08 |
| src/download.rs | 410 | cli | resources (Fable subagent) | clean |
| src/image.rs | 314 | core | pipeline (Fable subagent) | findings: PL-03, PL-09 |
| src/lib.rs | 73 | core | core/cli/config (Opus subagent) | findings: CO-15 |
| src/logbuf.rs | 276 | core | server+UI (Fable subagent) | findings: SV-08 |
| src/metrics.rs | 648 | core | server+UI (Fable subagent) | clean |
| src/model/classes.rs | 132 | model | pipeline (Fable subagent) | findings: PL-08 |
| src/model/detr.rs | 241 | model | pipeline (Fable subagent) | clean |
| src/model/mod.rs | 788 | model | pipeline (Fable subagent) | findings: PL-02, PL-05 |
| src/model/nms.rs | 109 | model | pipeline (Fable subagent) | findings: PL-01 |
| src/model/preprocess.rs | 329 | model | pipeline (Fable subagent) | findings: PL-04 |
| src/model/rtdetr.rs | 133 | model | pipeline (Fable subagent) | clean |
| src/model/yolo26.rs | 118 | model | pipeline (Fable subagent) | clean |
| src/model/yolo5.rs | 92 | model | pipeline (Fable subagent) | findings: PL-01, PL-02 |
| src/model/yolo8.rs | 94 | model | pipeline (Fable subagent) | findings: PL-01 |
| src/prep.rs | 413 | core | pipeline (Fable subagent) | findings: PL-04 |
| src/profile.rs | 882 | core | pipeline (Fable subagent) | clean |
| src/registry.rs | 1144 | core | pipeline (Fable subagent) | findings: PL-11 |
| src/registry/plan.rs | 390 | core | pipeline (Fable subagent) | findings: PL-11 |
| src/resources/actions.rs | 885 | resources | resources (Fable subagent) | findings: RS-16 |
| src/resources/catalog.rs 1-520 (types, consts, Platform/Flavor/Resource) | 520 | resources | resources (Fable subagent) | clean |
| src/resources/catalog.rs 1090-1244 (MODELS table) | 155 | resources | resources (Fable subagent) | clean |
| src/resources/catalog.rs 1244-1820 (export recipes, EXPORT_MODELS table) | 576 | resources | resources (Fable subagent) | clean |
| src/resources/catalog.rs 1820-2190 (export helpers, TOOLS, DEIMv2 source, lookups) | 370 | resources | resources (Fable subagent) | clean |
| src/resources/catalog.rs 2190-2500 (tests) | 310 | test | resources (Fable subagent) | clean |
| src/resources/catalog.rs 520-1090 (OpenVINO, ORT, CUDA wheels, model macros) | 570 | resources | resources (Fable subagent) | clean |
| src/resources/commands.rs | 445 | cli | resources (Fable subagent) | clean |
| src/resources/export.rs 1-450 (paths, locks, toolchain state, stages) | 450 | resources | resources (Fable subagent) | findings: RS-11 |
| src/resources/export.rs 1290-1760 (Exporter thread, export_now, messages) | 470 | resources | resources (Fable subagent) | findings: RS-11 |
| src/resources/export.rs 1760-2566 (fake + tests) | 806 | test | resources (Fable subagent) | clean |
| src/resources/export.rs 450-770 (seams: SystemRunner, readers, ManagerFetcher) | 320 | resources | resources (Fable subagent) | findings: RS-02, RS-07, RS-08, RS-04 |
| src/resources/export.rs 770-1290 (run_stages, install_outputs) | 520 | resources | resources (Fable subagent) | findings: RS-09 |
| src/resources/extract.rs 1-425 (dispatch, whitelists, zip/tgz drivers) | 425 | resources | resources (Fable subagent) | findings: RS-03, RS-13 |
| src/resources/extract.rs 425-900 (OpenVINO/ORT/NVIDIA selections and checks) | 475 | resources | resources (Fable subagent) | clean |
| src/resources/extract.rs 900-1617 (tests) | 717 | test | resources (Fable subagent) | clean |
| src/resources/manager.rs 1-560 (manifest, lock, jobs, status, Manager API) | 560 | resources | resources (Fable subagent) | findings: RS-04 |
| src/resources/manager.rs 1200-1545 (CLI progress, tests) | 345 | test | resources (Fable subagent) | clean |
| src/resources/manager.rs 560-1200 (worker loop, HTTP, run_job, put_in_place, download_part) | 640 | resources | resources (Fable subagent) | findings: RS-01, RS-05, RS-06, RS-10, RS-15 |
| src/resources/mod.rs | 33 | resources | resources (Fable subagent) | clean |
| src/resources/provision.rs | 515 | resources | resources (Fable subagent) | findings: RS-06 |
| src/resources/resolve.rs | 1086 | resources | resources (Fable subagent) | clean |
| src/resources/status.rs | 886 | resources | resources (Fable subagent) | findings: RS-14 |
| src/runner.rs | 584 | core | core/cli/config (Opus subagent) | findings: CO-05, CO-12 |
| src/server.rs | 634 | server | server+UI (Fable subagent) | findings: SV-01, SV-02, SV-04, SV-05, SV-09, SV-12, SV-15 |
| src/server/apply.rs | 414 | server | server+UI (Fable subagent) | clean |
| src/server/benchmark.rs 1-360 (page, template rows) | 360 | benchmark | server+UI (Fable subagent) | clean |
| src/server/benchmark.rs 1191-1967 (tests) | 777 | test | server+UI (Fable subagent) | clean |
| src/server/benchmark.rs 361-560 (config_view, saved cache, status) | 200 | benchmark | server+UI (Fable subagent) | clean |
| src/server/benchmark.rs 561-780 (start, settings, cancel, images_detail, image_file) | 220 | benchmark | server+UI (Fable subagent) | findings: SV-06 |
| src/server/benchmark.rs 781-1190 (apply, search, apply_threshold) | 410 | benchmark | server+UI (Fable subagent) | findings: SV-06 |
| src/server/browse.rs | 387 | server | server+UI (Fable subagent) | findings: SV-07, SV-10 |
| src/server/config_page.rs | 1266 | server | server+UI (Fable subagent) | findings: SV-02, SV-03 |
| src/server/detect.rs | 233 | server | server+UI (Fable subagent) | findings: SV-04, SV-08, SV-13 |
| src/server/devices.rs | 306 | server | server+UI (Fable subagent) | clean |
| src/server/live.rs | 493 | server | server+UI (Fable subagent) | findings: SV-05 |
| src/server/pages.rs | 905 | server | server+UI (Fable subagent) | findings: SV-09, SV-14 |
| src/server/resources.rs | 884 | server | server+UI (Fable subagent) | findings: SV-02 |
| src/server/test_util.rs | 318 | test | server+UI (Fable subagent) | clean |
| src/setup_onnxruntime.rs | 299 | cli | resources (Fable subagent) | clean |
| src/setup_openvino.rs | 177 | cli | resources (Fable subagent) | clean |
| src/startup.rs | 101 | core | core/cli/config (Opus subagent) | clean |
| src/sync.rs | 97 | core | pipeline (Fable subagent) | clean |
| src/system_info.rs | 112 | core | core/cli/config (Opus subagent) | findings: CO-13 |
| src/update.rs | 106 | server | resources (Fable subagent) | findings: RS-12 |
| src/worker.rs 1-520 (handles, spawn) | 520 | core | pipeline (Fable subagent) | findings: PL-01, PL-07 |
| src/worker.rs 1040-1551 (load, finish, tests) | 511 | core | pipeline (Fable subagent) | findings: PL-04, PL-10 |
| src/worker.rs 520-1040 (Pipeline, run loop, accept/fill/reply) | 520 | core | pipeline (Fable subagent) | findings: PL-06, PL-12 |
| templates/base.html | 43 | template | server+UI (Fable subagent) | clean |
| templates/benchmark.html | 951 | template | server+UI (Fable subagent) | findings: SV-09 |
| templates/config.html | 586 | template | server+UI (Fable subagent) | findings: SV-02 (forms post to admin routes without a token) |
| templates/logs.html | 129 | template | server+UI (Fable subagent) | clean |
| templates/message.html | 31 | template | server+UI (Fable subagent) | findings: SV-16 |
| templates/stats.html | 93 | template | server+UI (Fable subagent) | clean |
| templates/test.html | 158 | template | server+UI (Fable subagent) | clean |
| templates/welcome.html | 76 | template | server+UI (Fable subagent) | clean |
| tests/backend_cpu.rs | 117 | test | non-Rust+tests (Opus subagent) | findings: NR-23 |
| tests/backend_ort.rs | 306 | test | non-Rust+tests (Opus subagent) | findings: NR-14, NR-24 |
| tests/common/mod.rs | 397 | test | non-Rust+tests (Opus subagent) | findings: NR-24 |
| tests/data/bench/coco-dir/{annotations.json,a.jpg,b.jpg} | 3 files | test | benchmark-metrics (Opus subagent) | clean |
| tests/data/bench/manifest-set/{manifest.json,a.jpg,b.png} | 3 files | test | benchmark-metrics (Opus subagent) | clean |
| tests/data/bench/plain-dir/a.jpg | 1 file | test | benchmark-metrics (Opus subagent) | clean |
| tests/data/bench/yolo-dir/{classes.txt,labels/a.txt,a.jpg} | 3 files | test | benchmark-metrics (Opus subagent) | clean |
| tests/data/dog_bike_car.jpg | binary (164 KB) | asset | non-Rust+tests (Opus subagent) | clean (provenance not documented, see Not verified) |
| tests/enabled_models_http.rs | 476 | test | non-Rust+tests (Opus subagent) | findings: NR-14 |
| tests/families_http.rs | 100 | test | non-Rust+tests (Opus subagent) | findings: NR-14 |
| tests/integration_http.rs | 80 | test | non-Rust+tests (Opus subagent) | clean |
| tests/live_config.rs | 290 | test | non-Rust+tests (Opus subagent) | clean |
| tests/multi_model_http.rs | 375 | test | non-Rust+tests (Opus subagent) | findings: NR-14 |
| tests/pipeline.rs | 261 | test | pipeline (Fable subagent) | clean (ignored by default; see Tests) |
| tests/postprocess.rs | 630 | test | pipeline (Fable subagent) | clean |
| tests/provision.rs | 333 | test | resources (Fable subagent) | clean |
| tests/resources.rs | 1167 | test | resources (Fable subagent) | clean |
| tests/select.rs | 849 | test | non-Rust+tests (Opus subagent) | clean |

## Risk map

### Threat model

The port (default 32168, `bind_address` `0.0.0.0`) is reachable by Blue Iris on another LAN host, by
browsers on the same host, and by anything on the LAN. A LAN client can send multipart images up to
`BODY_LIMIT` (32 MiB) to the detection routes, hit `custom/list`, `/`, `/stats.json`, `/prometheus`,
`/v1/devices`, `/v1/status/updateavailable` and (today) `GET /v1/config`. Loopback clients, or every
client when `allow_remote_admin` is set (the shipped Docker config sets it), can additionally post forms
to the admin routes: write the config (including `save_image_path`, `log_path`, `cache_dir`), restart,
change the log level, start/remove downloads, add a model by any path, browse the filesystem, start and
cancel a model export, and run and apply benchmarks. There is no authentication and no CSRF or Origin
check, so a web page open in the administrator's browser can drive every admin action. The process can
download and extract pinned archives into the data root, run a user-triggered model export with `uv`
and Python (the only execution of downloaded code), load dynamic libraries from the runtime directories,
and write the config, cache, saved-image and log folders. On Windows the service runs as LocalSystem and
on macOS launchd as root, so the admin surface runs with full privileges on those platforms.

### Listeners, write paths, parsers of downloaded content, executed programs

- **Network listeners**: `src/server.rs:310` (`bind_address:port`, IPv4) and `:318` (`[::1]:port`,
  best effort). Bound once per process; generation restarts swap `Shared.current`.
- **Outbound HTTP**: `src/resources/manager.rs:753` (downloads; connect 30 s / read 60 s; no
  `https_only`, default redirect policy, RS-01), `src/update.rs:34` (GitHub release check, 10 s,
  cached 1 h, body unbounded RS-12).
- **Filesystem writes**: config file (`Config::save`, plain `fs::write`, CO-02) from every UI save and
  every CLI start; `benchmark.json` / `benchmark-preds.json` next to the config; `cache/` (OpenVINO
  cache, `cache/coreml/*.onnx` rewrite copies, ORT CoreML cache); `save_image_path` (`<stem>_od.jpg`,
  `<stem>.jpg`); `log_path` daily files; the download root (`openvino/`, `onnxruntime/<flavor>/`,
  `models/`, `tools/`, `bench/`, `.installed.json`, `.downloads.lock`, `*.part`, staging dirs).
- **Downloaded content parsed or extracted**: zip/tar.gz/wheel extraction with a name whitelist
  (`src/resources/extract.rs`); dataset manifests and images (`src/benchmark/images.rs`); GitHub
  release JSON (`src/update.rs`); ONNX model files (OpenVINO/ORT loaders and the protobuf wire parser in
  `src/backend/onnx_rewrite.rs`); class YAML; Hugging Face `preprocessor_config.json`.
- **Executed programs**: `src/resources/export.rs:587` runs the pinned `uv` and the Python it installs
  (hashes verified for uv, lock files, weights, DEIMv2 source before execution; child killed individually,
  not as a tree, RS-02); `src/system_info.rs:22/73/83` run `/usr/sbin/sysctl` and `powershell` (by
  search path, no timeout, CO-13). Dynamic libraries are loaded from the configured runtime directories;
  the CUDA probes load `cudart`/`cublas`/`cudnn`/`nvcuda` by bare name (BE-01).

### Unsafe blocks

All `unsafe` is in `src/backend/{detect,device,libs,ort}.rs` (47 lines, 24 blocks). Table from the
backend review:

| path:line | what it does | one-line justification / concern |
|---|---|---|
| detect.rs:613 | `CreateDXGIFactory1()` | COM factory creation; RAII interface. OK. |
| detect.rs:617 | `factory.EnumAdapters1(i)` | Valid factory; `DXGI_ERROR_NOT_FOUND` terminates. OK. |
| detect.rs:623 | `adapter.GetDesc1()` | Valid adapter; failure skipped. OK. |
| detect.rs:706 | `libs::open("nvcuda.dll" / "libcuda.so.1")` by bare name | Runs the driver's initializers; default search order (cwd before PATH on Windows). Only reached when DXGI already saw an NVIDIA adapter, so a real driver (System32) is normally present; residual risk noted in BE-01. Handle leaked on purpose (710). |
| detect.rs:716-733 | `transmute::<*const (), CuXxx>` of six symbols | Signatures match `cuda.h`; `"system"` ABI correct for both OSes. OK. |
| detect.rs:739-776 | `init`, `count`, `get`, `attr`, `name`, `bus`, `luid_fn` calls | Out-params are locals of documented sizes; `n.max(0)`; results checked. OK. |
| device.rs:598 | `unsafe impl Send for CallbackBox` | Only a raw pointer to the slot's own boxed `CallbackCtx` (`AtomicU64` + crossbeam `Sender`, both `Send + Sync`). Sound as long as the slot is never split; it moves as a unit. |
| device.rs:603-615 | `on_infer_done` dereferences `args` as `&CallbackCtx` | Context is `Box`ed in `OvSlot.ctx` with a stable address; `OvSlot` field order drops `request` (which waits for in-flight work) before `_callback` and `ctx`, and `OvBackend::drop` aborts first. Correct. Concern: relies on `ov_infer_request_free` waiting for the async job (true for OpenVINO 2.x `IAsyncInferRequest::~IAsyncInferRequest` -> `stop_and_wait`), unverified against the C wrapper. |
| device.rs:626 | `transmute_copy(r)` to read the `ov_infer_request_t*` out of `openvino::InferRequest` | Size asserted; layout of a single-field `repr(Rust)` struct is not formally guaranteed; `openvino = "0.11"` is not pinned exactly (BE-11). |
| device.rs:633-640 | `ov_get_last_err_msg` + `CStr::from_ptr` | Thread-local static string or null; null checked. OK. |
| device.rs:648 | `ov_infer_request_wait(raw)` | Live request borrowed `&mut`. OK; used because the crate's `wait_for` cannot distinguish timeout from done. |
| device.rs:665 | `ov_infer_request_set_callback(raw, &*cb)` | `ov_callback_t` boxed and kept in the slot (covers both "C copies the struct" and "C keeps the pointer"). OK. |
| libs.rs:129 | `std::env::set_var(OPENVINO_INSTALL_DIR, ..)` (Windows) | Win32 env block is internally locked; `std::env` readers are safe; C `getenv` reads the CRT copy. Acceptable; runs again in later registry generations. |
| libs.rs:272 | `Library::open(Some(path), RTLD_NOW \| RTLD_GLOBAL)` of TBB/libopenvino by absolute path | OpenVINO's own libraries from the configured dir; handles kept forever. OK. |
| libs.rs:308 | `std::env::set_var("PATH", ..)` (Windows) | As 129. Side effect: inherited by child processes (BE-15). |
| libs.rs:343-360 | `open(&path)`, `symbol::<SetProperty>`, `transmute_copy(core)`, variadic call, `ov_get_last_err_msg` (Apple arm64) | Works around openvino-sys calling a variadic C function with a fixed signature (would read garbage on Apple arm64). `loaded_openvino_c()` falls back to the finder's env search when nothing was loaded by path, which could in principle resolve a *different* `libopenvino_c.dylib` than the one openvino-sys loaded (then `ov_core_set_property` from library B gets a core from library A). Only when `OPENVINO_INSTALL_DIR`/`DYLD_LIBRARY_PATH` and the finder disagree; low likelihood, noted under "Not verified". |
| libs.rs:715 | `libloading::Library::new(path)` wrapper `open` | Contract forwarded to callers. OK. |
| libs.rs:731 | `lib.get(name)` wrapper `symbol` | Contract forwarded to callers. OK. |
| libs.rs:852 | `Library::open(Some(path), RTLD_NOW \| RTLD_GLOBAL)` of each CUDA lib, `mem::forget` | Absolute path from `cuda-libs`; by design loads *every* library-looking file in that directory (unknown families included). Acceptable given the directory sits next to the exe. |
| libs.rs:859 | `Library::load_with_flags(path, LOAD_WITH_ALTERED_SEARCH_PATH)` (Windows) | Absolute path; altered search resolves imports from the DLL's own directory first. Correct flag choice. |
| libs.rs:876 | `open(*n)` of `cudart64_12.dll`/`cublas64_12.dll`/`cudnn64_9.dll` (or `.so`) **by bare name**, handle dropped at once | Default search order (BE-01); repeated per `/v1/devices` and `/v1/resources` request (BE-02); unloading cudart/cuBLAS/cuDNN after their initializers ran is of unverified safety (CUDA documents that the runtime must not be unloaded while in use; here it was never used). |
| ort.rs:76-96 | `open(path)`, `symbol::<GetApiBase>`, `(*base).GetVersionString()`, `mem::forget(lib)` | ORT C API contract; absolute path chosen by the lookup; leak intentional. OK. |
| ort.rs:273 | `libs::open(p)` for each candidate incl. **bare name** `PathBuf::from(n)`; handle dropped | Same concern as libs.rs:876 (BE-01); runs once at startup per EP. |

### Non-test panic sites

Library and binary code outside `src/server/test_util.rs` and `#[cfg(test)]` modules (lead sweep: 23
`unwrap`/`expect`/`panic!`/`unreachable!` sites; the reviewers' tables below add index and slice
operations that depend on input):

| path:line | expression | reachable from | classification |
|---|---|---|---|
| plan.rs:217 | `failures.pop()...expect("one failure")` | never (guarded by `len() == 1`) | provably infallible |
| plan.rs:119,133 | `out[0]` | never (guarded by `out.len() == 1`) | provably infallible |
| select.rs:392 | `Device::new(..).expect("valid runtime/target combination")` | internal call sites only | provably infallible |
| select.rs:511 | `.next().expect("has Intel")` | guarded by `has_vendor(Intel)` | provably infallible |
| device.rs:321,464 | `inputs[image_idx]` | `image_idx` from `find_image_input` over the same slice | provably infallible |
| device.rs:485 | `img.shape[3]`, `img.shape[2]` | after `shape.len() != 4` check | provably infallible |
| device.rs:960 | `model.inputs[idx]` | `idx` from `image_index` over the same slice | provably infallible |
| device.rs:876,878,884,888 | `openvino::Tensor::get_data::<T>()` asserts slice alignment (dependency, openvino-0.11.0/src/tensor.rs:143) | every inference, for runtime-allocated output tensors | should be an error (in the dependency; OpenVINO allocations are aligned, unverified for all plugins) |
| device.rs:620-623, libs.rs:331-332 | `const _: () = assert!(size_of ...)` | compile time | provably infallible |
| detect.rs:548 | `nvidia.remove(pos)` | `pos` from `position` | provably infallible |
| detect.rs:739-776 | `name_buf.len() as c_int` etc. | fixed 256/64 | provably infallible |
| ort.rs:509,1035 | `inputs[image_idx]`, `self.info.inputs[image_idx]` | from `find_image_input`/`position` | provably infallible |
| ort.rs:529 | `img.shape[3]`, `[2]` | after rank check | provably infallible |
| ort.rs:1178 | `info.inputs[i]` | `i` from `extra_input_index` | provably infallible |
| ort.rs:1357 | `self.slots[slot]` | `slot in 0..len` | provably infallible |
| onnx_rewrite.rs:85,100 | `.try_into().expect("8 bytes")` / `"4 bytes"` | after `get(pos..pos+8/4)` | provably infallible |
| onnx_rewrite.rs:837-838 | `.expect("type checked above")` | after `scalar(dtype, alpha)` succeeded for the same dtype | provably infallible |
| onnx_rewrite.rs:1058 | `unreachable!("filtered above")` | op_type filtered by the `wanted` match | provably infallible |
| onnx_rewrite.rs:1106 | `n.outputs[0]` | after `outputs.len() == 1` | provably infallible |
| onnx_rewrite.rs:889 | `(dim + k - 1) / k` | crafted ValueInfo `dim_value` near `i64::MAX` with opset>=18 Split (`RewriteOptions::ALL` only) | should be an error (debug panic; release wraps) |
| onnx_rewrite.rs:917 | `start + size` | crafted `split` attribute values near `i64::MAX` (reachable with `COREML` options, opset<13) | should be an error (debug panic; release wraps) |
| onnx_rewrite.rs:969 | `lead.iter().product()` | crafted large dims, Flatten (`ALL` only) | should be an error (debug panic; release wraps) |
| src/model/mod.rs:211,223,235 | `unreachable!()` after `matches!` guard | never | provably infallible |
| src/model/mod.rs:371 | `ModelFamilyKind::Auto => unreachable!()` | never (`Auto` resolved above) | provably infallible |
| src/model/mod.rs:512-515 | `x.round() as usize` | model output (NaN -> 0) | should be an error (PL-02) |
| src/model/yolo26.rs:38 | `as_f32().unwrap_or_default()` | never None | provably infallible |
| src/model/yolo26.rs:63 | `dims.0 * dims.1` (wraps in release) | model shape | provably infallible (mismatch -> error) |
| src/model/yolo26.rs:113 | `cls.round() as usize` | model output | provably infallible (saturating) |
| src/model/classes.rs:66 | `.expect("bundled coco_classes.yaml is valid")` | compile-time asset | provably infallible |
| src/model/preprocess.rs:53 | `vec![0.0; 3*w*h]` (alloc abort) | model file static dims | should be an error (PL-04) |
| src/model/preprocess.rs:161-163 | `r_plane[start..end]` | geometry | provably infallible |
| src/image.rs:102,113 | `img.rgb[i..i+3]` | after length check | provably infallible |
| src/image.rs:34-35 | `image::load_from_memory` + `to_rgb8` (alloc up to ~1 GB) | request bytes | should be an error (PL-03) |
| src/worker.rs:437 | `crossbeam_channel::bounded(hard_cap)` (alloc abort) | config `worker_queue_size` | should be an error (PL-07) |
| src/worker.rs:772 | `debug_assert_eq!(i, i_tick)` | debug only | provably infallible |
| src/worker.rs:939,951,953 | `unreachable!`, `unwrap_or_default` | never | provably infallible |
| src/worker.rs:1000-1003,1020 | `p.slots[i]` | internal indices | provably infallible |
| src/worker.rs:1185 | `vec![114u8; w*h*3]` (alloc abort) | model file static dims | should be an error (PL-04) |
| src/registry.rs:523 | `workers[i]` | `resolve_default` over enabled models | provably infallible |
| src/registry.rs:755 | `s.by_name[&key]` | key checked at 746 under the reload mutex | provably infallible |
| src/registry.rs:709 | `s.workers[i]` (in `apply_live`) | `default_by_name` filters by `by_name` | provably infallible |
| src/resources/manager.rs:470 | `.expect("spawning the download thread")` | first `Manager::start` (service startup, CLI fetch) when the OS refuses a thread | should be a controlled abort (acceptable; could return `Result`) |
| src/resources/export.rs:1442 | `.expect("spawning the export thread")` | first Export click / `fetch --resource model:*` | should be a controlled abort (acceptable) |
| src/resources/catalog.rs:783 | `panic!("URL is not UTF-8")` in `const fn` | compile time only | provably infallible at run time |
| src/resources/manager.rs:282, 312 | `bytes * 100 / total` | progress text | provably infallible (`total > 0` guard; u64 headroom) |
| src/resources/status.rs:539 | `(bytes * 100 / total) as u8` | `/v1/resources` | provably infallible (guard; bytes ≤ total enforced by `download_part`) |
| src/resources/export.rs:467 | `a + ((b - a) * p / 100) as u8` | progress | provably infallible (`p.min(100)`, spans ≤ 100) |
| src/resources/export.rs:951 | `(b * 100).checked_div(t).unwrap_or(0) as u8` | uv download progress | infallible for b ≤ t (manager caps bytes at size) |
| src/resources/actions.rs:298, 333 | `files[0]` | Add-to-config | provably infallible (catalog parts non-empty; `model_paths` of an export has 2 entries) |
| src/resources/actions.rs:543 | `ok[0]` | custom model classes discovery | provably infallible (`len()==1` guard) |
| src/resources/actions.rs:435 | `s[1..s.len()-1]` | form path with quotes | provably infallible (len ≥ 2, ASCII quote at both ends) |
| src/resources/catalog.rs:2183 | `n[i..]`, `&n[..i]` | model lookup by user string | provably infallible (`is_char_boundary` filter) |
| src/setup_openvino.rs:116 | `resource.parts[0].url` | `setup-openvino` | provably infallible (catalog parts non-empty) |
| src/resources/resolve.rs:338 | `config.models[i]` (`ordered_models`) | every resolution | provably infallible (index from `effective_default_index` over the same vec) |
| src/resources/extract.rs:373 | `entry.path()?` | malformed tar entry name | should be an error (it is one, but loses the archive/entry context) |
| src/server.rs:364 | `std::panic::resume_unwind(p)` | any panic inside a `blocking` closure (e.g. a template bug) | should be an error (map to 500; there is no CatchPanic layer, so the connection is dropped instead — SV-15) |
| src/server.rs:366 | `panic!("blocking handler task did not finish: {e}")` | only when the runtime cancels the blocking task (shutdown) | should be a controlled abort (acceptable) |
| src/server/detect.rs:123 | `.as_millis().min(i32::MAX as u128) as i32` | every detection | provably infallible |
| src/server/detect.rs:128 | `start.elapsed().as_millis() as u64` | every successful detection | provably infallible (u128 -> u64 truncation only after 584 M years) |
| src/server/benchmark.rs:344-351 | `S[0]..S[3]`, `A[..]`, `U[..]` | `GET /benchmark` | provably infallible (constant arrays of 4) |
| src/server/browse.rs:133 | `UNITS[i]` | `GET /v1/fs/browse` | provably infallible (loop guard `i + 1 < UNITS.len()`) |
| src/metrics.rs:51 | `self.buckets[Self::bucket_index(ms)]` | every recorded sample | provably infallible |
| src/metrics.rs:166,170 | `self.dropped[reason as usize]` | every drop | provably infallible |
| src/logbuf.rs:146 | `entries.drain(..entries.len() - limit)` | `/logs.json` | provably infallible (guarded by `len > limit`) |
| src/benchmark/mod.rs:98 | `s[0]` | `Stats::from_samples` (empty returns early at 92) | provably infallible |
| src/benchmark/mod.rs:102 | `s[s.len() - 1]` | same | provably infallible |
| src/benchmark/mod.rs:112 | `sorted[rank.clamp(1, n) - 1]` (`clamp` panics when n == 0) | only `from_samples` today; `pub fn` | provably infallible today; should be an error for the pub API (BM-09) |
| src/benchmark/mod.rs:430 | `x.saturating_sub(..) * y.saturating_sub(..)` (usize mul) | `iou` via `run_agreement`/`compare` on `Detection::from(&ImagePred)` with a hand-edited huge bbox in benchmark.json (regrade then finish) | should be an error (debug panic, release wraps; BM-07) |
| src/benchmark/mod.rs:454-457 | `used_a[i]`, `used_b[j]`, `primary[i]`, `cpu[j]` | indices from `enumerate` | provably infallible |
| src/benchmark/mod.rs:463-464 | `primary.len() - matched` | matched ≤ min(len) | provably infallible |
| src/benchmark/mod.rs:948 | `all[0]` | `all.len() == 1` branch | provably infallible |
| src/benchmark/mod.rs:999 | `all[0]` | warm-up after `image_count() == 0` bail (831) | provably infallible |
| src/benchmark/mod.rs:1019 | `all.len() * repeat` | CLI `--repeat <huge>` (no upper bound) | should be an error (BM-04) |
| src/benchmark/mod.rs:1032-1033 | `Vec::with_capacity(repeat)` | CLI `--repeat 18446744073709551615` gives a capacity-overflow panic; ~1e9 gives an OOM abort | should be an error (BM-04) |
| src/benchmark/mod.rs:1051 | `last.expect("repeat >= 1")` | `repeat = self.repeat.max(1)` (907) | provably infallible |
| src/benchmark/mod.rs:1127-1134 | `self.samples[k]`, `e.0[0..3]` | fixed-size arrays, k < 4 | provably infallible |
| src/benchmark/mod.rs:1241 | `eval_preds[k]`, `per_image[k]` | built one per image in `timed_runs` | provably infallible |
| src/benchmark/mod.rs:1346 | `m.gt_matched[j]` | `match_image` returns one entry per gt of `gt_list` (= flattened `mapped`) | provably infallible (relies on metrics contract) |
| src/benchmark/mod.rs:1373-1379, 1433, 1462, 1468, 1485, 1525, 1581 | `all[k]`, `evals[k]`, `frames[k]`, `matched[k]`, `eval_preds[k]` | k from `0..all.len()` / `chosen` | provably infallible |
| src/benchmark/mod.rs:1674 | `run.per_image[k]` | one ImageRun per subset image, same order | provably infallible |
| src/benchmark/mod.rs:1790, 1811, 1817, 1829 | `selection.options[i]`, `planned[k]`, `slots[k]` | positions and ranges over the same vecs | provably infallible |
| src/benchmark/cli.rs:1654 | `a.matched + a.only_device.max(a.only_reference)` | agreement loaded from a hand-edited benchmark.json (also used by export.rs:95) | should be an error (saturating_add; BM-07) |
| src/benchmark/report.rs:108 | `r.per_image[k]` | length equality checked at 110 | provably infallible |
| src/benchmark/export.rs:517-518 | `ranking[0]` | caller checks `!ranking.is_empty()` (425) | provably infallible |
| src/benchmark/images.rs:710 | `p.0 as usize + 1` | user dataset `data.yaml` with key u64::MAX (wraps to 0 in release) | should be an error |
| src/benchmark/images.rs:711 | `vec![String::new(); len]` | user `data.yaml` with a huge index (alloc failure aborts the process) | should be an error |
| src/benchmark/images.rs:713 | `out[i as usize]` | after the wrap above: index out of bounds | should be an error |
| src/benchmark/images.rs:403 | `self.images[i * len / n]` | `--max-images` | provably infallible (i<n<len) |
| src/benchmark/images.rs:631 | `out[i]` | COCO JSON | provably infallible (index map of the same vec) |
| src/benchmark/metrics.rs:486 | `npos - tp` | every run | provably infallible (tp counts distinct non-ignored GT) |
| src/benchmark/metrics.rs:575 | `aps[0]` | every run | provably infallible (10 IoU thresholds) |
| src/benchmark/metrics.rs:727 | `img.gt[*j]` | every run | provably infallible (j from match_image over the same gt) |
| src/benchmark/metrics.rs:813, 903, 924 | `npos - tp` | sweep/search | provably infallible for `SweepImage::with_iou` output (fields are pub; only tests build them by hand) |
| src/benchmark/metrics.rs:898 | `self.steps[n - 1]` | sweep/search | provably infallible (n>0) |
| src/benchmark/metrics.rs:935 | `pts[(i*last+(max-1)/2)/(max-1)]` | chart decimation | provably infallible (max index = last) |
| src/benchmark/roc.rs:183 | `self.steps[n - 1]` | frame rates | provably infallible |
| src/benchmark/roc.rs:230 | `pts[...]` in `thin` | ROC curve | provably infallible |
| src/benchmark/roc.rs:89-103 | `while i < s.len()` with NaN | `auc()` with a NaN score (no current caller can pass one) | should be a controlled abort / error (hang, BQ-11) |
| src/benchmark/search.rs:154 | `self.images[k]` | search | provably infallible (k from enumerate) |
| src/benchmark/threshold.rs:272, 285, 297, 383 | `segments[..]`/`segs[..]` | exact pick | provably infallible (indices from the same slice) |
| src/cli.rs:25 | `.expect("PossibleValuesParser only passes known family names")` | `--family` parsing | provably infallible |
| src/config.rs:454 | `self.models[i]` (i from `position`) | default-model resolution | provably infallible |
| src/config_form.rs:187 | `on[i]` (i from `position` over the same vec length) | Models card POST | provably infallible |
| src/config_form.rs:208 | `c.models[i]` | Models card POST | provably infallible |
| src/config_merge.rs:465 | `target.models[i]` (i from `position`) | per-model reload rollback | provably infallible |
| src/runner.rs:279 | `self.listeners.take().expect(...)` | first `serve` only | provably infallible |
| src/runner.rs:301 | `self.server_task.as_mut().expect(...)` | after `serve` | provably infallible |
| src/config_store.rs:122 | `d.as_millis() as u64` | store creation | provably infallible |
| src/system_info.rs:68 | `kb * 1024` | profile detection (Linux) | provably infallible (needs MemTotal > 2^54 kB) |
| src/bin/blue_onyx_prism_service.rs:256 | `checkpoint += 1` (u32, every 0.5 s) | stop wait | provably infallible (68 years) |
| src/bin/test_blue_onyx_prism.rs:122 | `partial_cmp(b).unwrap()` | stats of elapsed times | provably infallible (finite values) |
| src/bin/test_blue_onyx_prism.rs:124,130,181 | `v[..]`, `outcomes[0]` | stats / single request | provably infallible (non-empty) |
| src/bin/test_blue_onyx_prism.rs:170 | `interval * (i / parallel) as u32` | `--interval-ms` × `--repeat` | should be an error (Duration overflow panic, usize→u32 truncation) |
| (via config) src/worker.rs:437 | `crossbeam_channel::bounded(hard_cap)` with `hard_cap = worker_queue_size` | config file, `--worker-queue-size`, `/config` form | should be an error (CO-03: upfront allocation of `cap` slots → OOM abort, or "capacity is too large" panic) |
| (via config) src/registry.rs:339 | `handle.request_timeout() + Duration::from_secs(1)` | `request_timeout_secs = 18446744073709551615` then a per-model reload | should be an error (CO-03: "overflow when adding durations") |

### Hot-path allocations and copies per request

Steady state, from the pipeline and server reviews: hyper reads the body into the 32 MiB-bounded
`Bytes`; the multipart parser copies the `image` part once into `Bytes` and builds a small `HashMap`
of text fields; one `oneshot`, one `PrepJob` (`Bytes` moved, zero-copy). On a prep thread: JPEG via
zune-jpeg, else the `image` crate (which allocates twice: `DynamicImage` then `to_rgb8`), into an
`RgbImage` buffer of `w*h*3` **allocated per request** (6 MB at 1080p; unbounded for non-JPEG inputs,
PL-03); letterbox/stretch writes into a pooled CHW `f32` tensor (reused per prep thread); RT-DETR adds a
small `extra_inputs` Vec. The backend copies the tensor into the infer request; outputs refill reused
`NamedOutput` buffers. Postprocess allocates a `Vec<Detection>` sized by the candidates above the
threshold (request-controlled `min_confidence` as low as `1e-6` makes that 25200/8400 candidates and an
O(N^2) NMS, PL-01), then one `String` per prediction and the JSON response. Optional image saving clones
the RGB buffer, encodes JPEG and writes on the worker thread. Bounded: body, tensor, outputs, queue depth
per model, log ring entry count. Not bounded: decoded image dimensions (non-JPEG), number of in-flight
connections and buffered bodies (SV-04), NMS candidate count, log ring bytes (SV-08).

### Findings ranked by severity

| Severity | Count |
|---|---|
| crash-or-data-loss | 7 |
| security-boundary | 19 |
| resource-exhaustion | 18 |
| correctness | 29 |
| operability | 33 |
| maintainability | 19 |
| **total** | **125** |

| # | ID | Severity | Title | Location | Phase |
|---|---|---|---|---|---|
| 1 | BM-01 | crash-or-data-loss | A corrupt or unloadable benchmark.json is silently overwritten, losing every other model's results | src/benchmark/cli.rs:1012, src/benchmark/service.rs:944, src/benchmark/report.rs:885-890 a | 1 |
| 2 | BQ-01 | crash-or-data-loss | YOLO `data.yaml` index map can abort the process or panic | src/benchmark/images.rs:705-714 | 1 |
| 3 | BQ-02 | crash-or-data-loss | A preds file that fails to load is silently overwritten, losing every other model's stored predictions | src/benchmark/search.rs:244-248 | 1 |
| 4 | CO-01 | crash-or-data-loss | `--model` with a relative config entry replaces the whole model list and writes it back | src/cli.rs:385 | 1 |
| 5 | CO-02 | crash-or-data-loss | Config writes are not atomic, so a crash or full disk mid-write leaves a truncated config | src/config.rs:395 | 1 |
| 6 | CO-03 | crash-or-data-loss | Unbounded numeric config fields crash the process (`worker_queue_size`; `request_timeout_secs` at `u64::MAX`) | src/config_form.rs:96-98 | 1 |
| 7 | PL-07 | crash-or-data-loss | `worker_queue_size` is not validated; a large value aborts at spawn | src/worker.rs:431-437, src/config.rs:594-636 | 1 |
| 8 | BE-01 | security-boundary | Bare-name DLL/so probes use the default loader search order (cwd before PATH on Windows) | src/backend/ort.rs:268 | 2 |
| 9 | BE-04 | security-boundary | CoreML rewrite cache in the shared temp dir is trusted on existence alone | src/backend/ort.rs:604-607 | 2 |
| 10 | BM-08 | security-boundary (low impact) | The Markdown report emits headings and paragraphs unescaped | src/benchmark/export.rs:776-782 | 2 |
| 11 | BQ-03 | security-boundary (latent; not reachable with untrusted data today) | `safe_file_name` accepts `:` (Windows drive-relative names and NTFS streams) | src/benchmark/images.rs:414-416 | 2 |
| 12 | CO-07 | security-boundary | `--allow-remote-admin` (and the other boolean flags) persist into the config file after a one-off run | src/cli.rs:331-333 | 2 |
| 13 | NR-01 | security-boundary | Docker defaults expose the unauthenticated admin surface to the network | deploy/docker-config.json:3, deploy/docker-compose.yml:12, Dockerfile:38 | 2 |
| 14 | NR-02 | security-boundary | Windows service runs as LocalSystem while the admin UI trusts every local (loopback) user | scripts/install_service.ps1:73 | 2 |
| 15 | NR-03 | security-boundary | macOS launchd daemon runs as root | deploy/com.blueonyx.prism.plist:4-21 | 2 |
| 16 | NR-04 | security-boundary | systemd install gives the service user write access to its own binary and libraries; no unit hardening | scripts/install_systemd.sh:17, deploy/blue-onyx-prism.service:5-16 | 2 |
| 17 | NR-05 | security-boundary | Workflow token permissions are broader than needed | .github/workflows/release.yml:15-16, .github/workflows/ci.yml:1-11 | 2 |
| 18 | PL-09 | security-boundary | Client-controlled image stem can name a Windows device file when `save_ref_image` is on | src/image.rs:200-215, src/worker.rs:1281-1294 | 2 |
| 19 | RS-01 | security-boundary | Download client follows redirects to any scheme; HTTPS-only is checked on the first URL only | src/resources/manager.rs:753-770 | 2 |
| 20 | RS-07 | security-boundary | Export child environment is blocklist-sanitized, not minimal | src/resources/export.rs:561-597 | 2 |
| 21 | RS-16 | security-boundary | "Add a model by path" reads and parses arbitrary files named by the admin form | src/resources/actions.rs:466-545 | 2 |
| 22 | SV-01 | security-boundary | No authentication, CSRF protection or Host/Origin validation on admin routes (same-host browser attack) | src/server.rs:149-187 | 2 |
| 23 | SV-03 | security-boundary | `save_image_path`, `log_path` and `cache_dir` from the Config page are not validated: arbitrary write locations | src/server/config_page.rs:640-685 | 2 |
| 24 | SV-05 | security-boundary | `GET /v1/config` is open to every client and returns the complete configuration | src/server.rs:142 | 2 |
| 25 | SV-09 | security-boundary (defence in depth) | No security response headers (nosniff, frame-ancestors/CSP, Referrer-Policy) and inline scripts/handlers everywhere | src/server.rs:188-191 | 2 |
| 26 | SV-10 | security-boundary (low; by design, documented) | Server-side file browser lists any directory the service account can read | src/server/browse.rs:207-216 | 2 |
| 27 | BE-02 | resource-exhaustion | `system_cuda_present()` loads and unloads cudart/cuBLAS/cuDNN on every `/v1/devices` and `/v1/resources` request | src/backend/libs.rs:872-884 | 3 |
| 28 | BE-06 | resource-exhaustion | Whole-model parse with up to ~20x memory expansion and no size cap | src/backend/onnx_rewrite.rs:1228 | 3 |
| 29 | BM-04 | resource-exhaustion | CLI `--repeat` has no upper bound (the service caps it at 1000); a huge value aborts or overflows | src/benchmark/cli.rs:148 | 3 |
| 30 | BM-11 | resource-exhaustion | Memory of a web UI run grows with images × devices × models until the final save | src/benchmark/service.rs:617-627 | 3 |
| 31 | BQ-04 | resource-exhaustion | Stored predictions file is unbounded and parsed whole on every search request | src/benchmark/search.rs:177-186 | 3 |
| 32 | BQ-05 | resource-exhaustion | Image bytes read without a size ceiling | src/benchmark/images.rs:224-230 | 3 |
| 33 | CO-06 | resource-exhaustion | Daily log files are never deleted | src/cli.rs:501 | 3 |
| 34 | NR-06 | resource-exhaustion | launchd log file grows without bound | deploy/com.blueonyx.prism.plist:17-20 | 3 |
| 35 | PL-01 | resource-exhaustion | Tiny `min_confidence` turns NMS into an O(N^2) CPU burn on the worker thread | src/worker.rs:400-407 | 3 |
| 36 | PL-03 | resource-exhaustion | Non-JPEG decode path has no dimension limit: a 1 MB PNG can cost ~1 GB and seconds per prep thread | src/image.rs:34-40 | 3 |
| 37 | PL-04 | resource-exhaustion | Model input size from the model file is unbounded; huge static dims abort the process | src/model/preprocess.rs:53, src/worker.rs:1185, src/prep.rs:23 & 132-138 | 3 |
| 38 | RS-02 | resource-exhaustion | Export child processes are killed individually; their descendants survive cancel, timeout and service stop | src/resources/export.rs:639-647 | 3 |
| 39 | RS-03 | resource-exhaustion | No decompressed-size ceiling during extraction (zip/tar bomb) | src/resources/extract.rs:287 | 3 |
| 40 | RS-05 | resource-exhaustion | Oversize bodies are retried forever and `Content-Length` is never checked against the pinned size | src/resources/manager.rs:1168-1175 | 3 |
| 41 | RS-08 | resource-exhaustion | Child output lines are buffered without a length bound | src/resources/export.rs:677-695 | 3 |
| 42 | RS-12 | resource-exhaustion | Release-check JSON body is unbounded and cached | src/update.rs:30, 49 | 3 |
| 43 | SV-04 | resource-exhaustion | No request-head, idle or in-flight limits on the listener; each upload is fully buffered | src/server.rs:334-356 | 3 |
| 44 | SV-08 | resource-exhaustion | Unauthenticated clients can fill the in-memory log ring with ~800 MiB via the unknown-model warning | src/server/detect.rs:106 | 3 |
| 45 | BE-03 | correctness | `OrtRuntime::full_name` names the wrong GPU when CUDA ordinals differ from the platform order | src/backend/ort.rs:571-583 | 4 |
| 46 | BE-05 | correctness | Unchecked i64 arithmetic on values parsed from the model in the Split/Flatten emitters | src/backend/onnx_rewrite.rs:917 | 1 |
| 47 | BE-07 | correctness | Device index accepted up to `u32::MAX` and cast to `i32` for ORT `device_id` | src/backend/spec.rs:276-281 | 4 |
| 48 | BE-09 | correctness | `device_class` for a bare `openvino:gpu` may classify a different GPU than the one `load_model` picked | src/backend/mod.rs:741-765 | 4 |
| 49 | BE-14 | correctness | OpenVINO `submit` bumps the sequence number before `infer_async`; a synchronous start failure may leave a live completion | src/backend/device.rs:981-986 | 4 |
| 50 | BM-02 | correctness | `--min-confidence` is not validated; NaN or inf is persisted as `null` and makes benchmark.json unloadable | src/benchmark/cli.rs:163, 318, 519 | 1 |
| 51 | BM-03 | correctness | `merge` drops all previous results on any HardwareSummary string difference, inconsistent with `same_gpus` | src/benchmark/report.rs:908 | 4 |
| 52 | BM-10 | correctness | Pseudo ground truth is capped at 100 reference detections per image | src/benchmark/mod.rs:1191-1192 | 4 |
| 53 | BQ-06 | correctness | Matching in crowd/ignore regions uses IoU, not pycocotools' intersection-over-detection | src/benchmark/metrics.rs:337-339 | 4 |
| 54 | BQ-07 | correctness | A pick that misses its precision/FPR target at every 0.01 step is marked `met: true` and is applied | src/benchmark/threshold.rs:367-394 | 4 |
| 55 | BQ-08 | correctness | A COCO JSON that matches no image silently turns the set into "annotated, zero objects" | src/benchmark/images.rs:640-655 | 4 |
| 56 | BQ-09 | correctness | YOLO label values are not validated | src/benchmark/images.rs:721-750 | 4 |
| 57 | CO-04 | correctness | The config file, CLI flags and on-disk edits are never validated, and only the `/config` form path validates | src/config.rs:380-387 | 1 |
| 58 | CO-08 | correctness | An external edit with the same length and mtime tick is missed, then overwritten by the next UI save | src/config_store.rs:263 | 4 |
| 59 | CO-09 | correctness | `pending()` does not report a removed per-model `inflight` (fields omitted when `None`) | src/config_merge.rs:531 | 4 |
| 60 | CO-10 | correctness | The CLI resolves a relative `log_path` from the config file against the cwd; the service resolves it against the exe dir | src/bin/blue_onyx_prism.rs:348 | 4 |
| 61 | CO-14 | correctness | Test client: `--interval-ms` delays compound with `buffered`, and a large interval panics | src/bin/test_blue_onyx_prism.rs:170 | 5 |
| 62 | NR-07 | correctness | workflow_dispatch can rewrite an already-published release | .github/workflows/release.yml:13, :144 | 4 |
| 63 | NR-08 | correctness | Ultralytics exports are not checked against PyTorch, although the docs say every export is | scripts/export_model.py:364-371 | 4 |
| 64 | NR-09 | correctness | README Docker instructions put models where the compose file never mounts them | README.md:618-621, deploy/docker-compose.yml:23 | 4 |
| 65 | NR-10 | correctness | README says "nothing downloaded is executed" | README.md:120 | 4 |
| 66 | NR-11 | correctness | Stale or incomplete README statements | README.md:184, :188, :231, :283, :413-414, :651 | 4 |
| 67 | PL-02 | correctness | Non-finite model outputs reach the response as `(0,0,0,0)` boxes and `"confidence": null` | src/model/mod.rs:504-515, src/model/yolo5.rs:62, src/api.rs:58 | 1 |
| 68 | PL-05 | correctness | `family: auto` misdetects a 1-class YOLOv5 export as YOLO26 | src/model/mod.rs:396-406 | 4 |
| 69 | PL-08 | correctness | Class index maps with gaps shift every later label | src/model/classes.rs:32-49 | 4 |
| 70 | RS-09 | correctness | `install_outputs` moves two files and writes the manifest non-atomically | src/resources/export.rs:1236-1275 | 4 |
| 71 | RS-15 | correctness | Range resume carries no validator; a changed upstream file becomes a permanent, non-retried failure | src/resources/manager.rs:1115 | 4 |
| 72 | SV-13 | correctness (low) | A detection request with a non-multipart content type is answered without draining the body | src/server/detect.rs:72-79 | 4 |
| 73 | SV-14 | correctness | Models-card per-model threshold accepts any finite number (e.g. 1.5 or -1) | src/server/live.rs:441-466 | 4 |
| 74 | BE-08 | operability | Config `openvino_dir` silently ignored when `OPENVINO_INSTALL_DIR` is set | src/backend/libs.rs:85-99 | 4 |
| 75 | BE-10 | operability | A dead ORT session thread disables its slot for good; `Drop` can block on a hung session | src/backend/ort.rs:1296-1304 | 4 |
| 76 | BE-13 | operability | `openvino:gpu.N` plan only checks that some GPU exists | src/backend/plan.rs:150-169 | 4 |
| 77 | BE-15 | operability | Windows `PATH` mutation is inherited by child processes | src/backend/libs.rs:288-312 | 4 |
| 78 | BM-05 | operability | A cancel issued right after start can be lost (flag reset after the state lock is released) | src/benchmark/service.rs:351-369 | 4 |
| 79 | BM-06 | operability | The dataset download wait has no deadline, and model compile is not cancellable | src/benchmark/service.rs:529-568 | 4 |
| 80 | BQ-10 | operability | Dataset resolution: one bad folder fails the run, id collisions silently drop a set | src/benchmark/images.rs:857-858, 891-893 | 4 |
| 81 | CO-05 | operability | Windows service first start: default config has no models, so no HTTP UI and a 5-second retry loop forever | src/cli.rs:403-416 | 4 |
| 82 | CO-11 | operability | Config handling before logging is set up loses its messages, and a bad `log_path` aborts the CLI server | src/bin/blue_onyx_prism.rs:346-348 | 4 |
| 83 | CO-12 | operability | Shutdown cannot be forced: a second signal is ignored and the drains have no upper bound | src/bin/blue_onyx_prism.rs:372-391 | 4 |
| 84 | CO-13 | operability | Windows memory detection spawns `powershell` by search path with no timeout | src/system_info.rs:83-91 | 4 |
| 85 | NR-12 | operability | The release checklist's CUDA check runs on a GPU the pinned Windows CUDA build cannot use | docs/RELEASE_CHECKLIST.md:13-20 | 4 |
| 86 | NR-13 | operability | No release gate for the primary target's Intel iGPU path | docs/RELEASE_CHECKLIST.md:6-21, .github/workflows/ci.yml:47-56 | 4 |
| 87 | NR-14 | operability | CI's "real model" tests skip every non-IPcam family and the lazy and RT-DETR/CoreML paths | .github/workflows/ci.yml:43-44, :59-66 | 4 |
| 88 | NR-15 | operability | smoke.sh accepts a detection with zero predictions; provider pattern is an unescaped regex | scripts/ci/smoke.sh:53-58, :64-66 | 4 |
| 89 | NR-16 | operability | install_service.ps1 ignores sc.exe failures and passes the account password on the command line | scripts/install_service.ps1:61-63, :71, :73, :75-77 | 4 |
| 90 | NR-17 | operability | Release archives and the binary ship third-party code without its license notices | .github/workflows/release.yml:93-98, LICENSE:25-49, src/resources/extract.rs:121 | 4 |
| 91 | NR-18 | operability | Docker image has no HEALTHCHECK; builder image tag floats | Dockerfile:9, :20, :36-38 | 4 |
| 92 | NR-19 | operability | Bind-mounted config and cache directories may not be writable by the image's `openvino` user (unverified) | Dockerfile:34-38, deploy/docker-compose.yml:24-27 | 4 |
| 93 | NR-20 | operability | The weekly URL check reports nowhere a human reliably sees; jobs have no timeouts | .github/workflows/check-urls.yml:8-10, :25 | 4 |
| 94 | NR-21 | operability | Install/uninstall scripts are incomplete for upgrade and removal | scripts/install_systemd.sh:27, scripts/install_launchd.sh | 4 |
| 95 | PL-06 | operability | A hung inference stalls the model forever: `unstick` converts a lost completion into a blocking `collect` | src/worker.rs:1018-1030, src/worker.rs:1248 | 4 |
| 96 | PL-11 | operability | A reload can block the config-save path behind a compiling model | src/registry/plan.rs:48-51, src/registry.rs:627-631 | 4 |
| 97 | RS-04 | operability | A dead download worker leaves jobs "queued" forever: `install_all`, the exporter and blocked models hang | src/resources/manager.rs:642-658 | 4 |
| 98 | RS-06 | operability | Service downloads retry without bound; a blocked model never reaches a terminal state | src/resources/provision.rs:128 | 4 |
| 99 | RS-10 | operability | Service stop can block on an uninterruptible verify/extract of a large resource | src/resources/manager.rs:551-557 | 4 |
| 100 | RS-11 | operability | CLI export has no cancellation and waits on the export lock forever | src/resources/export.rs:1677-1683 | 4 |
| 101 | RS-14 | operability | `/v1/resources` rows do heavy filesystem and hardware probing per request | src/resources/status.rs:504-690 | 4 |
| 102 | SV-06 | operability | Benchmark API handlers do blocking file I/O, multi-MB JSON parsing, config writes and CPU work on the 2-thread async runtime | src/server/benchmark.rs:570-614 | 3 |
| 103 | SV-07 | operability | `/v1/fs/browse` walks the filesystem and probes every Windows drive letter on the async thread | src/server/browse.rs:96-105 | 3 |
| 104 | SV-11 | operability (low) | `live.js` polls forever and swallows non-JSON/403 answers; `waitRestart` never gives up | assets/live.js:19-26 | 4 |
| 105 | SV-12 | operability (documentation) | Admin guard trusts `ConnectInfo` only; reverse proxies on loopback grant admin to everyone | src/server.rs:196-205 | 4 |
| 106 | SV-15 | operability | Handler panics close the connection without a response; the `blocking` doc claims a 500 | src/server.rs:358-369 | 1 |
| 107 | BE-11 | maintainability | Layout assumptions on `openvino` crate types are enforced by size only, with a non-exact version pin | src/backend/device.rs:618-627 | 5 |
| 108 | BE-12 | maintainability | Stale phase comment on `OrtProbe` | src/backend/select.rs:144-145 | 5 |
| 109 | BM-07 | maintainability | Unchecked integer arithmetic on values deserialized from benchmark.json | src/benchmark/cli.rs:1654, src/benchmark/mod.rs:151 and 430 | 5 |
| 110 | BM-09 | maintainability | `pub fn percentile` panics on an empty slice | src/benchmark/mod.rs:109-113 | 5 |
| 111 | BQ-11 | maintainability (latent) | NaN scores hang `auc()` and break the binary searches of the sweeps | src/benchmark/roc.rs:89-103 | 5 |
| 112 | BQ-12 | maintainability (latent) | Non-finite grade fields round-trip as `null` | src/benchmark/grade.rs:226-229 | 5 |
| 113 | BQ-13 | maintainability | Search request: duplicate model names and CSV escaping | src/benchmark/search.rs:504, 827-831, 887-893 | 5 |
| 114 | CO-15 | maintainability | No crate-level lint policy, so "no unsafe outside backend/" is not enforced | src/lib.rs:1-50 | 5 |
| 115 | CO-16 | maintainability | The confidence-threshold rule differs by path: 0 is allowed globally and in `validate_models`, but rejected per model by the benchmark apply | src/config_form.rs:120 | 5 |
| 116 | CO-17 | maintainability | Stale docs: the merge rule in config.rs and the README config table | src/config.rs:2-3 | 5 |
| 117 | NR-22 | maintainability | Unused dependencies in Cargo.toml | Cargo.toml:58 | 5 |
| 118 | NR-23 | maintainability | Stale guidance in CLAUDE.md, an agent definition and a test comment | CLAUDE.md:116-117, .claude/agents/porter.md:14-15, tests/backend_cpu.rs:110 | 5 |
| 119 | NR-24 | maintainability | Test fixtures leak temp directories and share an ORT cache across runs | tests/common/mod.rs:119-123, tests/backend_ort.rs:58, :232-245 | 5 |
| 120 | NR-25 | maintainability | Advisory ignores are undated; the toolchain pin lacks components | deny.toml:6-11, rust-toolchain.toml:1-2 | 5 |
| 121 | NR-26 | maintainability | build_bench_sets.py validation uses `assert`; Pillow is unpinned despite the determinism claim | scripts/build_bench_sets.py | 5 |
| 122 | PL-10 | maintainability | `device` RwLock bypasses the poison-recovering helpers | src/worker.rs:620-625, src/worker.rs:1123-1125 | 5 |
| 123 | PL-12 | maintainability | The pipeline run loop is only exercised by an ignored integration test | src/worker.rs:633-800 | 5 |
| 124 | RS-13 | maintainability | `write_entry` temp name and tar path errors lose information | src/resources/extract.rs:284 | 5 |
| 125 | SV-16 | maintainability | `message.html` interpolates HTML-escaped values into a JavaScript string literal | templates/message.html:13 | 5 |


## Findings in detail

### BM-01  A corrupt or unloadable benchmark.json is silently overwritten, losing every other model's results
- Severity: crash-or-data-loss
- Location: src/benchmark/cli.rs:1012, src/benchmark/service.rs:944, src/benchmark/report.rs:885-890 and 906-907
- Trigger: `benchmark.json` fails to parse. Causes include a truncated file, a hand edit, an older or newer program version whose required field is missing or renamed (for example a `RunResult` without `cache` or `image_size`), or a non-finite float written as `null` (see BM-02). The next CLI `--all-devices` or web UI run then saves.
- Evidence:
  ```rust
  let mut merged = BenchmarkResults::merge(BenchmarkResults::load_or_warn(&path), results);
  ...
  pub fn merge(previous: Option<Self>, new: Self) -> Self {
      let Some(prev) = previous else { return new };
  ```
  `load_or_warn` maps the parse error to a `tracing::warn!` and `None`, so the merge keeps only this run's models and `save` replaces the file. The measured `auto` device order of every other model disappears with it.
- Proposed fix: distinguish "missing" from "unparseable". On a parse error, move the file aside (`benchmark.json.bad-<timestamp>`) and surface a warning in the run status or CLI output before saving. Also consider deserializing `models` element-wise (`Vec<serde_json::Value>` then per-model `from_value`) so one bad entry does not discard the others.
- Phase: 1 (crash safety)

### BQ-01  YOLO `data.yaml` index map can abort the process or panic
- Severity: crash-or-data-loss
- Location: src/benchmark/images.rs:705-714
- Trigger: a user dataset dir (config `benchmark.datasets` `dir:` / UI) with `data.yaml` `names: {100000000000: x}` (alloc of 1e11 Strings -> allocation failure -> process abort, not caught by the job's catch_unwind) or `names: {18446744073709551615: x}` (`as usize + 1` wraps to 0 in release -> `out[i]` out of bounds panic; overflow panic in debug). A key like 50_000_000 just allocates ~1.2 GB.
- Evidence: `let len = pairs.last().map_or(0, |p| p.0 as usize + 1);` / `let mut out = vec![String::new(); len];` / `out[i as usize] = n;`
- Proposed fix: reject indices above a sane bound (e.g. > 10_000 or > 4x the number of entries) with an error naming the file; use `checked_add`.
- Phase: 1 (crash safety)

### BQ-02  A preds file that fails to load is silently overwritten, losing every other model's stored predictions
- Severity: crash-or-data-loss
- Location: src/benchmark/search.rs:244-248
- Trigger: `benchmark-preds.json` unreadable or unparsable (hand edit, truncated by a full disk outside the rename path, or a future `StoredRun` field without `serde(default)`); the next benchmark of any one model writes a file holding only that model's runs. The search page then reports "no stored predictions" for the rest until each is re-benchmarked.
- Evidence: `let prev = StoredPreds::load(&path).map_err(|e| tracing::warn!("{e:#}")).ok().flatten();` then `StoredPreds::merge(prev, merged, new).save(&path)`
- Proposed fix: on a load error other than NotFound, rename the bad file aside (`benchmark-preds.json.bad-<ts>`) before saving, and surface the warning in the run's warnings; add `serde(default)` to optional `StoredRun` fields.
- Phase: 1 (crash safety)

### CO-01  `--model` with a relative config entry replaces the whole model list and writes it back
- Severity: crash-or-data-loss
- Location: src/cli.rs:385 (verified); the absolutization is at src/bin/blue_onyx_prism.rs:21-22 and src/cli.rs:217
- Trigger: the config has `"path": "models/IPcam-general.onnx"`, the relative form that `download-models --add-to-config` writes (download.rs:75-82 strips the exe dir). The user then runs the documented `blue-onyx-prism --model models/IPcam-general.onnx --family yolo5` from the install directory. The CLI path becomes `<cwd>/models/IPcam-general.onnx`, which never equals the relative `m.path`, so `found` stays false and `config.models` is replaced by one unnamed entry. Every other model, with its names, thresholds, devices and `default_model` target, is lost, and the result is saved immediately (cli.rs:220).
- Evidence:
  ```
  for m in config.models.iter_mut().filter(|m| &m.path == model) {
  ...
  if !found {
      config.models = vec![ModelConfig {
  ```
- Proposed fix: compare resolved paths, `config.data_path(&m.path) == *model` (and canonicalize both when they exist). Consider appending instead of replacing, or replacing in memory only without writing it back.
- Phase: 1

### CO-02  Config writes are not atomic, so a crash or full disk mid-write leaves a truncated config
- Severity: crash-or-data-loss
- Location: src/config.rs:395 (verified); callers at src/cli.rs:220 (every CLI start), src/cli.rs:413, src/config_store.rs:288, src/bin/blue_onyx_prism.rs:259 and :287
- Trigger: power loss, a kill, or a full disk during `std::fs::write`, which opens with O_TRUNC and then writes. Because `resolve_config` rewrites the file on every CLI start, each boot of a systemd/launchd unit opens this window. Next start: `Config::load` fails to parse, the CLI exits with "fix the config file", and the Windows service retries every 5 s with no HTTP. The models list and all settings are gone. `ConfigStore::update` has the same window for every UI save, and its error path leaves memory intact but the file broken.
- Evidence:
  ```
  let text = serde_json::to_string_pretty(&named)?;
  std::fs::write(path, text).with_context(|| format!("writing config {}", path.display()))
  ```
- Proposed fix: write to `<path>.tmp` in the same directory, `sync_all`, then `rename` over the target (on Windows, `MoveFileExW` with REPLACE_EXISTING via `std::fs::rename`). Skip the startup write-back when the merged config equals the loaded one.
- Phase: 1

### CO-03  Unbounded numeric config fields crash the process (`worker_queue_size`; `request_timeout_secs` at `u64::MAX`)
- Severity: crash-or-data-loss
- Location: src/config_form.rs:96-98 (verified, no upper bound); src/cli.rs:367-369; src/config.rs:474-476; crash sites src/worker.rs:432-437 and src/registry.rs:339 (verified)
- Trigger:
  - `worker_queue_size: 1000000000`, set in the file, with `--worker-queue-size`, or through the `/config` form, which accepts any usize. crossbeam's array flavor allocates `cap` slots up front (crossbeam-channel `flavors/array.rs:127`). Values near `usize::MAX` panic with "bounded channel capacity is too large", and large values abort on OOM.
  - Because the field is "needs restart", the crash happens at the next `POST /config/restart` or at process start, and since the value is persisted, every start crashes. A crash loop under systemd or the service's auto-restart.
  - Separately, `request_timeout_secs: 18446744073709551615` passes `max(1)`, and `request_timeout() + Duration::from_secs(1)` in `retire_worker` panics with "overflow when adding durations" on the next per-model reload.
- Evidence:
  ```
  if let Some(v) = get("worker_queue_size") {
      c.worker_queue_size = num("worker_queue_size", v)?;
  }
  ...
  let (sender, receiver) = crossbeam_channel::bounded::<WorkItem>(hard_cap);
  ```
- Proposed fix: add range checks in one `Config::validate` that is used by load, form, CLI merge and disk reload, for example:
  - `worker_queue_size <= 1024`
  - `request_timeout_secs` in 1..=3600
  - `intra_threads`, `preprocess_threads` and `ort_sessions` at most 256, and `inflight` at most `MAX_INFLIGHT`
  - `gpu_index <= 64`

  Each error message should name the key and the range. Also clamp `worker_queue_size` at `spawn_worker_after`.
- Phase: 1

### PL-07  `worker_queue_size` is not validated; a large value aborts at spawn
- Severity: crash-or-data-loss
- Location: src/worker.rs:431-437, src/config.rs:594-636 (no check)
- Trigger: config `"worker_queue_size": 1000000000` (editable via the config page, `config_merge.rs:171`): `crossbeam_channel::bounded(cap)` allocates the slot array eagerly (~80 B x cap) -> allocation failure abort while spawning the first worker.
- Evidence: `let hard_cap = if cfg.queue_size > 0 { cfg.queue_size } else { MAX_AUTO_QUEUE }; let (sender, receiver) = crossbeam_channel::bounded::<WorkItem>(hard_cap);`
- Proposed fix: reject or clamp `worker_queue_size` to 1..=1024 in `validate_config` and clamp again in `spawn_worker_after`.
- Phase: 1

### BE-01  Bare-name DLL/so probes use the default loader search order (cwd before PATH on Windows)
- Severity: security-boundary
- Location: src/backend/ort.rs:268 (`candidates.push(PathBuf::from(n))`, loaded at :273), src/backend/libs.rs:876 (`open(*n)`), src/backend/detect.rs:706 (`open("nvcuda.dll")`)
- Trigger: Windows host with an NVIDIA GPU where `cudart64_12.dll`/`cublas64_12.dll`/`cudnn64_9.dll` are not in the ORT dir or `cuda-libs` (e.g. before `nvidia-cuda-libs` is downloaded). The probe calls `LoadLibraryExW(name, 0, 0)` (libloading 0.8/0.9 `Library::new` = `load_with_flags(filename, 0)`, verified at libloading-0.9.0/src/os/windows/mod.rs:69-70), whose default search visits the process working directory before `PATH`. A DLL of that name planted in the cwd (a console run from a user folder) is loaded and its `DllMain` runs with the service's privileges. `nvcuda.dll` is normally found in System32 first, so detect.rs is low risk; the cudart/cublas/cudnn names are *not* in System32.
- Evidence: `candidates.push(PathBuf::from(n));` / `match unsafe { libs::open(p) }` (ort.rs:268-273); `cuda_core_libs(cfg!(windows)).iter().all(|n| { match unsafe { open(*n) } ...` (libs.rs:873-876).
- Proposed fix: probe only by absolute path in the known directories (ORT dir, `cuda-libs`, and `PATH` entries enumerated explicitly), or on Windows call `load_with_flags(name, LOAD_LIBRARY_SEARCH_DEFAULT_DIRS | LOAD_LIBRARY_SEARCH_USER_DIRS)` / set `SetDefaultDllDirectories` once at startup so bare-name loads never consult the cwd.
- Phase: 2

### BE-04  CoreML rewrite cache in the shared temp dir is trusted on existence alone
- Severity: security-boundary
- Location: src/backend/ort.rs:604-607 (`std::env::temp_dir().join("blue-onyx-prism-coreml")`), src/backend/onnx_rewrite.rs:1236 (cache hit = file exists and non-empty)
- Trigger: macOS with `cache_dir` unset (None) on a multi-user machine. The cache path `<tmp>/blue-onyx-prism-coreml/<stem>-<sha16>-v1.onnx` is predictable from the model file (its stem and SHA-256 are not secret). Another local user pre-creates that file; the service then loads their model instead of the operator's. With `cache_dir` set, the directory is under the data root and the risk is the usual "who can write next to the exe".
- Evidence: `if std::fs::metadata(&path).is_ok_and(|m| m.is_file() && m.len() > 0) { return Ok(Cached::Rewritten { path, ... created: false }) }` (onnx_rewrite.rs:1236-1242).
- Proposed fix: never fall back to the shared temp dir (use `<data root>/cache/coreml`, or skip the rewrite when there is no cache dir); and/or verify a cache hit (store the expected SHA-256 of the rewritten bytes in a sidecar or re-derive it, since the rewrite is deterministic and cheap).
- Phase: 2

### BM-08  The Markdown report emits headings and paragraphs unescaped
- Severity: security-boundary (low impact)
- Location: src/benchmark/export.rs:776-782
- Trigger: a model name, dataset id (`dir:<path>`), error string (`Not benchmarked: {e}`, `failed: {e}` from runtime or model-file errors) or recommendation text containing raw HTML, Markdown syntax or newlines. Example: a model named `<img src=x onerror=…>`, or an ORT error message with a newline followed by `# …`. Markdown viewers that render inline HTML (VS Code preview, many static-site generators; GitHub sanitizes) show it, and newlines break the document structure. The test at export.rs:893 asserts the raw `<b>m</b>` survives. Table cells escape only `|` and `\n`, and a preceding `\` defeats the `|` escape. The HTML output is fully escaped and is not affected.
- Evidence:
  ```rust
  let esc = |s: &str| s.replace('|', "\\|").replace('\n', " ");
  let _ = writeln!(out, "{} {}\n", "#".repeat(s.level as usize), s.heading);
  for p in &s.paragraphs { let _ = writeln!(out, "{}\n", p); }
  ```
- Proposed fix: run headings and paragraphs through an escaper that replaces newlines and escapes `<`, `>`, `&` (and leading `#`, `*`, `_`, `` ` ``, `[`), and escape `\` before `|` in cells.
- Phase: 2 (security)

### BQ-03  `safe_file_name` accepts `:` (Windows drive-relative names and NTFS streams)
- Severity: security-boundary (latent; not reachable with untrusted data today)
- Location: src/benchmark/images.rs:414-416 (used at 364 and src/benchmark/report.rs:700)
- Trigger: a manifest file name `C:x.jpg` on Windows: `dir.join("C:x.jpg")` replaces the base path (prefix without root), so the file is read from the current directory of drive C; `a.jpg:stream` names an alternate data stream. Today names come only from embedded manifests (tested safe) and report.rs additionally requires the name to be a known image, so nothing untrusted reaches it.
- Evidence: `!f.is_empty() && !f.contains(['/', '\\']) && f != "." && f != ".." && !f.contains('\0')`
- Proposed fix: also reject `:` (or require `Path::new(f).components()` to be exactly one `Normal` component with no prefix) so the helper is safe if a downloaded/user manifest ever feeds `from_manifest`.
- Phase: 2 (security)

### CO-07  `--allow-remote-admin` (and the other boolean flags) persist into the config file after a one-off run
- Severity: security-boundary
- Location: src/cli.rs:331-333 (verified); write-back at src/cli.rs:220
- Trigger: `blue-onyx-prism --allow-remote-admin` run once for a test. The merged config is saved with `"allow_remote_admin": true`, and later plain starts keep serving admin routes (config editor, file browser, restarts) to the LAN. No CLI flag turns it off. `--force-cpu` and `--save-ref-image` behave the same way.
- Evidence:
  ```
  if cli.allow_remote_admin {
      config.allow_remote_admin = true;
  }
  ```
- Proposed fix: do not write security-relevant flags back (apply them in memory only), or make them `Option<bool>` (`--allow-remote-admin=false`) and log at warn level when remote admin is on.
- Phase: 2

### NR-01  Docker defaults expose the unauthenticated admin surface to the network
- Severity: security-boundary
- Location: deploy/docker-config.json:3, deploy/docker-compose.yml:12, Dockerfile:38
- Trigger: `docker compose up` with the shipped files. Any LAN host can reach `http://<host>:32168/config` and use the other admin routes. These include `/v1/fs/browse` (lists folders and model/YAML names of the container filesystem), config writes, `/v1/models/add`, resource downloads and benchmark runs.
- Evidence: docker-config.json:3 `"allow_remote_admin": true,`; compose:12 `- "32168:32168"`. README:625 recommends `-p 127.0.0.1:32168:32168` but the compose file does not do it. REVIEW_PLAN design target: "Users do not administer the service from another machine".
- Proposed fix: Ship `allow_remote_admin: false` and document admin over `docker exec` or an SSH tunnel. Alternatively, add an allow-list setting (e.g. trust only the Docker bridge gateway, which is what host-originated connections come from in the default iptables mode) instead of opening admin to every peer. Detection must stay on all interfaces, so publishing the port on loopback is not an option for remote Blue Iris.
- Phase: 2

### NR-02  Windows service runs as LocalSystem while the admin UI trusts every local (loopback) user
- Severity: security-boundary
- Location: scripts/install_service.ps1:73 (also :8, :15)
- Trigger: any non-admin local user on the Blue Iris PC opens `http://127.0.0.1:32168/config`. Loopback is trusted, and there is no authentication.
  - They can set `save_image_path` / `log_path` (both form fields in templates/config.html), which makes SYSTEM write files into any directory.
  - They can start a model export, which runs downloaded uv/Python/PyTorch as SYSTEM in `<install dir>/tools/`.
  - They can add an arbitrary model file by path.
  - If the install dir is user-writable (e.g. extracted under the user's profile, or a folder created at `C:\` that inherits `Authenticated Users: Modify`), they can replace the exe or DLLs the service loads.
  - Exploitation through the server routes is unverified here: those routes are reviewed by other reviewers.
- Evidence: install_service.ps1:73 `sc.exe create $ServiceName binPath= $binPath start= auto DisplayName= "$DisplayName" obj= LocalSystem`. The script never checks or sets the ACL of `$InstallDir`.
- Proposed fix: Default to a virtual service account (`obj= "NT SERVICE\BlueOnyxPrismService"`) with Modify granted only on the data dirs, and fall back to LocalSystem only when asked. In the script, reset the install dir ACL to Administrators/SYSTEM full and Users read+execute.
- Phase: 2

### NR-03  macOS launchd daemon runs as root
- Severity: security-boundary
- Location: deploy/com.blueonyx.prism.plist:4-21 (no `UserName` key), scripts/install_launchd.sh:23-27
- Trigger: `sudo scripts/install_launchd.sh`. The daemon, its runtime downloads, the dlopen of downloaded OpenVINO/ORT dylibs and any user-triggered model export (uv + Python) all run as root. Any local user reaches the loopback-trusted admin UI.
- Evidence: the plist has `Label`, `ProgramArguments`, `WorkingDirectory`, `RunAtLoad`, `KeepAlive`, `StandardOutPath`, `StandardErrorPath` and no `UserName`/`GroupName`.
- Proposed fix: Create a `_blueonyx` role user in install_launchd.sh (dscl), chown only the data dirs to it, and add `<key>UserName</key><string>_blueonyx</string>` to the plist.
- Phase: 2

### NR-04  systemd install gives the service user write access to its own binary and libraries; no unit hardening
- Severity: security-boundary
- Location: scripts/install_systemd.sh:17, deploy/blue-onyx-prism.service:5-16
- Trigger: after install, a compromise of the service process (it parses untrusted images and model files) can replace `/opt/blue-onyx-prism/blue-onyx-prism` and the dlopened `openvino/` and `onnxruntime/` libraries. That compromise then persists across restarts.
- Evidence: `chown -R blueonyx:blueonyx "$INSTALL_DIR"`. The unit has only `User=`, `Group=`, `SupplementaryGroups=`, with no `NoNewPrivileges`/`ProtectSystem`/`ProtectHome`/`PrivateTmp`.
- Proposed fix: Keep the binary root-owned. Put downloads, config, cache and models in a `StateDirectory=blue-onyx-prism` (or a `download_dir`) owned by the service user. Add `NoNewPrivileges=yes`, `ProtectSystem=strict`, `ReadWritePaths=<state dirs>`, `ProtectHome=yes`, `PrivateTmp=yes`.
- Phase: 2

### NR-05  Workflow token permissions are broader than needed
- Severity: security-boundary
- Location: .github/workflows/release.yml:15-16, .github/workflows/ci.yml:1-11 (no `permissions:`)
- Trigger:
  - release.yml: the three build jobs compile code and run downloads with a `contents: write` token.
  - ci.yml: push-to-main runs get the repository's default token permissions, which may be write on older repo settings (unverified for this repo).
- Evidence: release.yml:15 `permissions:` / :16 `contents: write` at workflow level. ci.yml has no permissions block.
- Proposed fix: Set top-level `permissions: contents: read` in both workflows, and grant `contents: write` only on the `release` job.
- Phase: 2

### PL-09  Client-controlled image stem can name a Windows device file when `save_ref_image` is on
- Severity: security-boundary
- Location: src/image.rs:200-215, src/worker.rs:1281-1294
- Trigger: `save_image_path` + `save_ref_image: true` and a multipart file name `CON.jpg` / `NUL.jpg` / `COM1.jpg`: the stem is `CON`, and `std::fs::write(dir.join("CON.jpg"))` on Windows opens the console device (`CON.<ext>` is still reserved for non-verbatim paths). Unverified on this machine (macOS); the `_od.jpg` suffix makes the annotated file safe, the reference copy is not. Over-long or `:`/`*` stems only fail with a logged warning.
- Evidence: `let stem = Path::new(name).file_stem()...; let ref_path = dir.join(format!("{stem}.jpg")); std::fs::write(&ref_path, ...)`
- Proposed fix: sanitize the stem to `[A-Za-z0-9._-]`, cap its length (e.g. 64), and reject Windows reserved names (CON, PRN, AUX, NUL, COM1-9, LPT1-9) by falling back to the UUID name.
- Phase: 2

### RS-01  Download client follows redirects to any scheme; HTTPS-only is checked on the first URL only
- Severity: security-boundary
- Location: src/resources/manager.rs:753-770 (verified: `fn http_client` 753, `fn check_url` 762)
- Trigger: a pinned host (GitHub release downloads and Hugging Face both answer with a 302 to a CDN) redirects to an `http://` location, or a compromised/MITM'd first hop injects a redirect. reqwest 0.13's default policy follows up to 10 hops with no scheme restriction; the client sets neither `https_only(true)` nor a `redirect::Policy`.
- Evidence:
  ```rust
  reqwest::Client::builder()
      .user_agent(concat!("blue-onyx-prism/", env!("CARGO_PKG_VERSION")))
      .connect_timeout(Duration::from_secs(30))
      .read_timeout(Duration::from_secs(60))
      .build()
  ```
  and `check_url` only inspects `part.url`. Integrity is still protected by the SHA-256 check (a tampered body is deleted), so the impact is a plaintext hop plus a non-retried "hash mismatch" denial, not code execution.
- Proposed fix: build the client with `.https_only(true)` in production (keep the loopback `http://` exception for tests via a `redirect::Policy::custom` that refuses non-`https` targets unless the host is loopback) and cap hops explicitly (`Policy::limited(5)`). Unit-test a 302 to `http://`.
- Phase: 2

### RS-07  Export child environment is blocklist-sanitized, not minimal
- Severity: security-boundary
- Location: src/resources/export.rs:561-597 (verified `fn sanitized` 561, `env_remove` 595)
- Trigger: the service's own environment carries `LD_PRELOAD`/`LD_LIBRARY_PATH`/`DYLD_INSERT_LIBRARIES`, `SSL_CERT_FILE`/`REQUESTS_CA_BUNDLE`/`CURL_CA_BUNDLE`, `XDG_*`, `TMPDIR`, `OMP_*`/`MKL_*`, a non-standard `PATH`, or `*_PROXY`: all reach `uv` and `python`. Proxies are probably wanted (uv must download wheels), but CA-bundle and loader variables change what the pinned toolchain trusts and loads. Python itself is hardened (`-I`, offline vars) and uv runs `--no-config`, so this is defense in depth, not an open hole.
- Evidence:
  ```rust
  ["PYTHON", "UV_", "PIP_", "CONDA_", "YOLO_", "HF_", "HUGGINGFACE_", "TRANSFORMERS_", "TORCH_", "VIRTUAL_ENV", "__PYVENV"]
      .iter().any(|p| k.starts_with(p))
  ```
- Proposed fix: `Command::env_clear()` and pass an explicit allowlist (`PATH`, `HOME`/`USERPROFILE`, `SystemRoot`, `TEMP`/`TMP`/`TMPDIR`, `LANG`/`LC_*`, `HTTP(S)_PROXY`/`NO_PROXY`, `SSL_CERT_FILE` only if the operator opted in) plus the explicit `cmd.env` entries; keep the test `sanitized_variables_and_package_progress` as the inverse check.
- Phase: 2

### RS-16  "Add a model by path" reads and parses arbitrary files named by the admin form
- Severity: security-boundary
- Location: src/resources/actions.rs:466-545 (verified `fn add_custom_model` 466, `given.is_dir()` 472, `load_class_names(&p)` 529)
- Trigger: a POST with `path=/etc/passwd` (or any absolute path) and `classes=/path/to/file`: the handler stats arbitrary paths, lists arbitrary directories (`pick_model_in_dir` returns the file names it found in the 409 message), and parses any readable file as YAML (`load_class_names`), with distinct error texts for "does not exist" vs parse failure. By design (CLAUDE.md: Browse... + add-by-path), and the route is loopback/admin only, but with `allow_remote_admin` it becomes a filesystem-existence and directory-listing oracle for anyone who can reach the port.
- Evidence:
  ```rust
  let given = config.data_path(Path::new(raw));
  let file = if given.is_dir() { pick_model_in_dir(&given)? } ...
  crate::model::classes::load_class_names(&p)
  ```
- Proposed fix: accept it as an admin feature but (a) do not echo directory contents in error messages when the directory is outside the data root, (b) cap the YAML read size, and (c) document in the config help that `allow_remote_admin` exposes this. Optionally restrict to the roots `GET /v1/fs/browse` already exposes.
- Phase: 2

### SV-01  No authentication, CSRF protection or Host/Origin validation on admin routes (same-host browser attack)
- Severity: security-boundary
- Location: src/server.rs:149-187 (admin router), src/server.rs:196-205 (`admin_allowed`), src/server.rs:210-249 (`admin_only`) (verified)
- Trigger: the admin's browser (on the service host, or any host when `allow_remote_admin` is on, which
  is the shipped Docker default, README:623) visits any web page that contains
  `<form method=post action="http://127.0.0.1:32168/config"><input name=allow_remote_admin value=on>
  <input name=access_settings value=1>...` and auto-submits it. `application/x-www-form-urlencoded`
  POSTs need no CORS preflight; the TCP peer is 127.0.0.1, so `admin_allowed` passes. The same works
  for `/config/restart`, `/config/models` (disable all but one model, change devices), `/v1/resources/
  download` + `confirm_large=1` (multi-GB downloads), `/v1/resources/remove` (delete model/runtime files),
  `/v1/models/add` (register any readable file as a model), `/v1/resources/export` + `confirm_large=1`
  (download and run the export toolchain), `/v1/benchmark` (start a benchmark against an arbitrary
  `dir:` folder, e.g. the user's pictures), and `/config` with `save_image_path`/`log_path` set to any
  directory (see SV-03). With DNS rebinding (attacker hostname re-resolving to 127.0.0.1; the Host
  header is never checked) the attacker additionally **reads** every admin response: `/v1/fs/browse`
  (whole filesystem listing), `/logs.json`, `/config`, `/v1/resources`.
- Evidence: `pub(super) fn admin_allowed(peer: Option<SocketAddr>, allow_remote_admin: bool) -> bool {
  allow_remote_admin || peer.is_some_and(|a| match a.ip() { ... is_loopback() ... }) }`; no middleware
  looks at `Origin`, `Sec-Fetch-Site`, `Host` or a token; forms in `templates/config.html:22,65,104,179`
  carry no token; `live.js:369` posts with `fetch(url, {method:"POST", body: new URLSearchParams(data)})`.
- Proposed fix: reject state-changing admin requests whose `Sec-Fetch-Site` is `cross-site` or whose
  `Origin`/`Referer` host does not match the request `Host` (and require `Host` to be the bound
  address/`localhost`); additionally accept only `Accept: application/json` + a custom header such as
  `X-Requested-With` for the JSON API paths, which forces a CORS preflight. Document that
  `allow_remote_admin` exposes an unauthenticated admin API to the whole network.
- Phase: 2

### SV-03  `save_image_path`, `log_path` and `cache_dir` from the Config page are not validated: arbitrary write locations
- Severity: security-boundary
- Location: src/server/config_page.rs:640-685 (`config_submit` -> `live::server_form_edit` ->
  `config_form::apply_config_form`), with the verbatim stores at src/config_form.rs:111-112 and 137-141 (verified)
- Trigger: `POST /config` with `save_image_path=C:\Windows\System32\drivers\etc` (or `/etc`,
  `\\server\share`, `../../..`), `image_settings=1`, `save_ref_image=on`. Every later detection writes a
  JPEG (and the unannotated copy) into that directory as the service account (SYSTEM for the Windows
  service). Combined with SV-01 this is reachable from a web page. `log_path` likewise selects where the
  rolling log file is created after a process restart; `cache_dir` selects where OpenVINO writes cache
  blobs.
- Evidence: `if let Some(v) = get("save_image_path") { c.save_image_path = opt_path(v); }` with no
  check; `ConfigView` (config_page.rs:41,46-47) only round-trips the strings.
- Proposed fix: validate path fields in `apply_config_form` (or in `validate_config`): require an
  absolute path or one under the data root after `canonicalize` of the existing parent, refuse UNC /
  verbatim prefixes and `..` components, and refuse system directories; show the resolved path on the
  page. At minimum log at `warn` when a save changes these fields.
- Phase: 2

### SV-05  `GET /v1/config` is open to every client and returns the complete configuration
- Severity: security-boundary
- Location: src/server.rs:142 (`.route("/v1/config", get(live::config_json))` in the **open** router), src/server/live.rs:316-336 (`payload`) (verified)
- Trigger: `curl http://<host>:32168/v1/config` from any machine on the network (no
  `allow_remote_admin` needed). The JSON contains `"config"` (every field: `bind_address`,
  `allow_remote_admin`, `download_dir`, `models_dir`, `cache_dir`, `log_path`, `save_image_path`, all
  model file paths, benchmark `dir:` folders), `"path"` (the config file's absolute path), `"fileError"`,
  the last 20 change-log entries (with sources and summaries) and `forms.server.view`. It also calls
  `sync_config`, so a remote GET can be what triggers the reload of models after an on-disk config edit.
- Evidence: `"config": cfg, ... "path": state.config_path.display().to_string(), ... "changes":
  state.config.changes(CHANGES_SHOWN)`; comment at server.rs:134 "Open to every client: ... read-only
  JSON".
- Proposed fix: move `/v1/config` into the admin router (the only consumers are the admin pages via
  `live.js` and `message.html`, which are themselves admin-only), or split it: an open, minimal
  `{revision, generation, epoch, restartNeeded, pendingCount}` for the restart poller and the full
  payload under admin.
- Phase: 2

### SV-09  No security response headers (nosniff, frame-ancestors/CSP, Referrer-Policy) and inline scripts/handlers everywhere
- Severity: security-boundary (defence in depth)
- Location: src/server.rs:188-191 (router has no `SetResponseHeaderLayer`), src/server/pages.rs:608-641 (static assets), templates/benchmark.html:23,108 (`onsubmit="return false"`), every template's inline `<script>` (verified)
- Trigger: any page can be framed by another origin (clickjacking of the Restart / Remove / Export
  buttons on the Config page once SV-01 is fixed by Origin checks; buttons inside an iframe still send
  same-origin requests); responses carry no `X-Content-Type-Options`; a future XSS has no CSP to blunt
  it, and introducing one requires moving every inline script and the two inline handlers to
  `live.js`/nonces.
- Evidence: `open.merge(admin).fallback(fallback).layer(DefaultBodyLimit::max(BODY_LIMIT))` — the only
  layer; `grep -rn nosniff|Content-Security-Policy|referrer src/` returns nothing.
- Proposed fix: add a small middleware (`map_response`) setting `X-Content-Type-Options: nosniff`,
  `X-Frame-Options: DENY` / `Content-Security-Policy: frame-ancestors 'none'; default-src 'self';
  img-src 'self' data: blob:; script-src 'self' 'nonce-…'`, `Referrer-Policy: no-referrer`,
  `Cache-Control: no-store` on JSON; move inline scripts into per-page static files and drop the two
  inline `onsubmit` attributes.
- Phase: 2

### SV-10  Server-side file browser lists any directory the service account can read
- Severity: security-boundary (low; by design, documented)
- Location: src/server/browse.rs:207-216 (`start` = any absolute path), 1-3 (module doc) (verified)
- Trigger: with `allow_remote_admin: true` (Docker default), `GET /v1/fs/browse?path=/` or
  `?path=C:\Users` from any LAN client returns every folder name (and model/YAML files) recursively on
  demand; on Windows the service runs as SYSTEM, so this includes other users' profile directories.
  Combined with DNS rebinding it is readable from a web page even without remote admin (SV-01).
- Evidence: `let start = if raw.is_empty() { ... } else { cfg.data_path(Path::new(raw)) };
  let canon = start.canonicalize()...` — no `starts_with` against a root set.
- Proposed fix: keep the design but gate it: only allow roots the page offers (data root, models dir,
  home, drives) **or** document clearly that remote admin = remote directory listing; refuse paths whose
  canonical form is outside `roots()` unless an `allow_browse_anywhere` setting is on. Fixing SV-01 is
  the prerequisite.
- Phase: 2

### BE-02  `system_cuda_present()` loads and unloads cudart/cuBLAS/cuDNN on every `/v1/devices` and `/v1/resources` request
- Severity: resource-exhaustion
- Location: src/backend/libs.rs:872-884; called from src/resources/resolve.rs:1043-1044 inside `detect_installed`, which `GET /v1/devices` (src/server/devices.rs:132) and the resources status page (src/resources/status.rs:389, src/server/resources.rs:638) call per request.
- Trigger: any machine with an NVIDIA GPU: each config-page/device-dropdown load performs three `LoadLibrary`/`FreeLibrary` (or `dlopen`/`dlclose`) cycles; `cublas64_12.dll` is on the order of 100 MB, and every load runs the libraries' static initializers. Loading then immediately unloading the CUDA runtime is not a documented-safe pattern (unverified whether cudart/cuDNN tolerate `dlclose` after their constructors ran; it has never been used, which makes it likely benign, but it is repeated).
- Evidence: `cuda_core_libs(cfg!(windows)).iter().all(|n| { match unsafe { open(*n) } { Ok(_) => true, ...` (libs.rs:873-877) — the `Library` is dropped at the end of the match arm.
- Proposed fix: cache the result once per process in a `OnceLock<bool>` (or alongside `CUDA_PRELOADED`), and when a result must be fresh, test for file existence in the enumerated search directories instead of loading.
- Phase: 3

### BE-06  Whole-model parse with up to ~20x memory expansion and no size cap
- Severity: resource-exhaustion
- Location: src/backend/onnx_rewrite.rs:1228 (`std::fs::read(src)`), :70-112 (`parse_fields` builds one 40-byte `Field` per encoded field), :205-221 (`rep_int` expands packed varints to `Vec<i64>`), :372-402 (`Tensor::ints` expands int32 raw data 2x)
- Trigger: CoreML path only. A legitimately large `.onnx` (e.g. a 1-2 GB RF-DETR L) or a crafted file whose messages are many 2-byte fields: the file is read fully, every node/attribute/tensor message is split into `Field`s (bounded by ~20x the message size), attribute ints are expanded 8x, and the rewritten copy is held alongside the original. Bounded by a constant factor of the file size, so not unbounded, but a few GB of transient memory for a large model on a laptop.
- Evidence: `let bytes = std::fs::read(src)...; let r = match rewrite(&bytes, opts)` (1228-1229); `out.push(Field { num, value, raw: &buf[start..pos] })` for every field (105-109).
- Proposed fix: skip the rewrite above a size threshold (e.g. 512 MB; such models should use external data, which is already skipped), and parse `Tensor` attributes lazily (only the fields the rewrite needs) rather than materializing every field of every weight tensor in `Graph::parse`.
- Phase: 3

### BM-04  CLI `--repeat` has no upper bound (the service caps it at 1000); a huge value aborts or overflows
- Severity: resource-exhaustion
- Location: src/benchmark/cli.rs:148 (no bound; only `== 0` is rejected at 488), src/benchmark/mod.rs:1019, 1032-1033
- Trigger: `benchmark --repeat 18446744073709551615` gives a `Vec::with_capacity` capacity-overflow panic. `--repeat 2000000000` tries 16 GB of allocations (an OOM abort, not an error). `all.len() * repeat` overflows (a debug panic; release wraps the progress total).
- Evidence:
  ```rust
  let total_iters = all.len() * repeat;
  ...
  let mut totals = Vec::with_capacity(repeat);
  let mut infers = Vec::with_capacity(repeat);
  ```
- Proposed fix: validate `--repeat` against `service::MAX_REPEAT` (and `--warmup` against `MAX_WARMUP`) in `check_args`. Use `checked_mul` for `total_iters` and do not pre-size by `repeat`.
- Phase: 3 (resources/load)

### BM-11  Memory of a web UI run grows with images × devices × models until the final save
- Severity: resource-exhaustion
- Location: src/benchmark/service.rs:617-627 (`finished` keeps full results), src/benchmark/mod.rs:1830 (re-clone of all device results per device)
- Trigger: the web UI with a `dir:` dataset and `max_images = 0` (all images; `max_images` is unbounded in `from_form`) over many models and devices. Each `RunResult` holds `per_image` (≤100 preds per image) plus uncapped `eval_preds` at confidence 0.05. All are retained for every finished model until `save` at the end, and `sweep` clones the whole slot list after every device.
- Evidence:
  ```rust
  Ok(r) => finished.push(r),
  ...
  result.devices = slots.iter().flatten().cloned().collect();
  ```
- Proposed fix: merge and save each model as it finishes (keeping only summaries in memory), or cap images per run in `RunRequest::validate`. In `sweep`, build the partial view without cloning `per_image`/`eval_preds` (for example by emitting `summary()` from the slots directly).
- Phase: 3 (resources/load)

### BQ-04  Stored predictions file is unbounded and parsed whole on every search request
- Severity: resource-exhaustion
- Location: src/benchmark/search.rs:177-186 (load), 201-225 (merge keeps all devices of all models); callers src/server/benchmark.rs:980, 1045 (async, no `server::blocking`)
- Trigger: many models x devices x all three built-in sets (384 images) with high-recall models at 0.05 (tens to hundreds of boxes per image): tens of MB of JSON, read and deserialized per `POST /v1/benchmark/threshold-search` and per options request, on one of the two tokio worker threads that also serve `/v1/vision/detection`.
- Evidence: `let text = match std::fs::read_to_string(path) { ... }; serde_json::from_str(&text)`; no cap on runs or predictions per image.
- Proposed fix: run the load + search in `server::blocking`, cache the parsed file keyed by mtime, and cap stored predictions per image (e.g. top 100 per class, matching `MAX_DETS`).
- Phase: 3 (resources/load)

### BQ-05  Image bytes read without a size ceiling
- Severity: resource-exhaustion
- Location: src/benchmark/images.rs:224-230
- Trigger: a user dataset dir containing a multi-GB `.jpg`/`.png` (or a FIFO/symlink to a huge file named `*.jpg`): `std::fs::read` loads it fully before any decode limit applies.
- Evidence: `Some(p) => std::fs::read(p).map(Cow::Owned)`
- Proposed fix: check `metadata().len()` against a limit (the server's request body limit is a natural choice) and fail that image with a message.
- Phase: 3 (resources/load)

### CO-06  Daily log files are never deleted
- Severity: resource-exhaustion
- Location: src/cli.rs:501 (verified)
- Trigger: a long-running service with `log_path` set. One file is created per day and none is removed; at `debug` (the service default, cli.rs:409) every request logs. The disk fills over months.
- Evidence:
  ```
  let appender = tracing_appender::rolling::daily(dir, "blue_onyx_prism.log");
  ```
- Proposed fix: use `tracing_appender::rolling::Builder::new().rotation(Rotation::DAILY).filename_prefix("blue_onyx_prism").filename_suffix("log").max_log_files(N)` with N configurable (for example 14). Consider making the service default `info`.
- Phase: 3

### NR-06  launchd log file grows without bound
- Severity: resource-exhaustion
- Location: deploy/com.blueonyx.prism.plist:17-20
- Trigger: a long-running daemon at `info` level (or `debug`), or a crash loop under `KeepAlive`, appends to `/usr/local/blue-onyx-prism/blue-onyx-prism.log` forever. There is no newsyslog entry.
- Evidence: `<key>StandardOutPath</key><string>/usr/local/blue-onyx-prism/blue-onyx-prism.log</string>` (and the same for stderr). install_launchd.sh adds no `/etc/newsyslog.d/` file.
- Proposed fix: Have install_launchd.sh install a `/etc/newsyslog.d/blue-onyx-prism.conf` rotation entry, or set `log_path` in the config (daily rolling files) and point stdout and stderr at `/dev/null`.
- Phase: 3

### PL-01  Tiny `min_confidence` turns NMS into an O(N^2) CPU burn on the worker thread
- Severity: resource-exhaustion
- Location: src/worker.rs:400-407 (`effective_threshold`), src/model/nms.rs:19-35, src/model/yolo5.rs:55-78, src/model/yolo8.rs:74-91
- Trigger: any client posts `min_confidence=0.000001` (or the config sets `confidence_threshold: 0`, allowed by `validate_models`) to a YOLOv5 model (25200 anchors) or YOLOv8 (8400).
- Evidence: `if min_confidence > 0.0 && min_confidence.is_finite() { min_confidence }` ... `for i in 0..n { for j in (i + 1)..n { ... iou(...) >= iou_threshold`
- Proposed fix: floor the effective threshold (e.g. 0.01) and cap the pre-NMS candidate set like Ultralytics (`max_nms` 30000 -> top 3000 by score) and the output (`max_det` 300); postprocessing runs on the model's only thread, so one such request stalls every queued request of that model.
- Phase: 3

### PL-03  Non-JPEG decode path has no dimension limit: a 1 MB PNG can cost ~1 GB and seconds per prep thread
- Severity: resource-exhaustion
- Location: src/image.rs:34-40
- Trigger: a client posts a highly compressible 11000 x 11000 RGBA PNG (< 32 MB body limit); `image`'s default `Limits` only caps a single allocation at 512 MiB, `to_rgb8()` then copies into a second 363 MB buffer, and `fast_image_resize` processes the full frame.
- Evidence: `let dynimg = image::load_from_memory(bytes).context("decoding image")?; let rgb = dynimg.to_rgb8();`
- Proposed fix: use `ImageReader::with_guessed_format()` + `limits(Limits { max_image_width: Some(8192 or 16384), max_image_height: ..., max_alloc: Some(256 MiB) })`, and `into_rgb8()` (no copy when already RGB8); apply the same side cap to the zune path for symmetry.
- Phase: 3

### PL-04  Model input size from the model file is unbounded; huge static dims abort the process
- Severity: resource-exhaustion
- Location: src/model/preprocess.rs:53, src/worker.rs:1185, src/prep.rs:23 & 132-138
- Trigger: a custom model (UI "Add a model by path", `--model`) with static image input e.g. `[1,3,16384,16384]`: `Preprocessor::new` allocates 3.2 GB of f32, the warm-up `gray` 805 MB, and the pool may retain up to 32 such tensors per length. `hf_preprocessor_size` clamps 32..=4096 but `device.rs:478-485` / `ort.rs:517` only reject `d <= 0`.
- Evidence: `tensor: vec![0.0; 3 * input_w as usize * input_h as usize],` and `let gray = vec![114u8; in_w as usize * in_h as usize * 3];`
- Proposed fix: validate `input_size` at load (sides in 32..=4096, area <= 4096^2) and fail the model with a clear message; bound `TensorPool` by total bytes (or `MAX_POOLED_PER_LEN` sized from `slots + prep threads`) and drop entries for lengths no model uses.
- Phase: 3

### RS-02  Export child processes are killed individually; their descendants survive cancel, timeout and service stop
- Severity: resource-exhaustion
- Location: src/resources/export.rs:639-647 (verified `child.kill()` at 639 and 644)
- Trigger: user clicks Cancel, a stage timeout fires, or the service stops while `uv python install` / `uv pip install` / `python export_model.py` runs. `uv` spawns helper processes and PyTorch export spawns worker processes; `Child::kill` only signals the direct child, so the grandchildren keep running (CPU, disk, network, and holding the venv files so a re-run / "Remove export toolchain" can fail). The test `system_runner_streams_fails_times_out_and_cancels` only uses `sh -c "sleep 30"`, where `sh` execs directly, so it does not exercise this.
- Evidence:
  ```rust
  if cancel.load(Ordering::Relaxed) {
      let _ = child.kill();
      let _ = child.wait();
      return Err(anyhow!(Cancelled));
  }
  ```
- Proposed fix: on Unix put the child in its own process group (`std::os::unix::process::CommandExt::process_group(0)`) and kill the group (`libc::kill(-pid, SIGKILL)` via `nix`, inside `backend/`-style unsafe or a crate); on Windows assign the child to a Job Object with `JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE` (or use `CREATE_NEW_PROCESS_GROUP` + `TerminateJobObject`). Add a test with `sh -c "sleep 30 & wait"`.
- Phase: 3

### RS-03  No decompressed-size ceiling during extraction (zip/tar bomb)
- Severity: resource-exhaustion
- Location: src/resources/extract.rs:287 (verified `std::io::copy(reader, &mut f)` in `write_entry`), callers 312-346 and 354-392
- Trigger: a deflate/gzip stream that expands to many GB for a whitelisted entry. For catalog downloads this requires defeating the SHA-256 pin (the catalog is trusted), but `setup-openvino --archive <file>` / `Job::local_archives` installs any user-supplied archive without a hash, and a future pin mistake has no second line of defense. The file name whitelist bounds the *number* of files but not their size.
- Evidence:
  ```rust
  let n = std::io::copy(reader, &mut f).with_context(|| format!("writing {}", out.display()))?;
  ```
  with no `.take()` on `reader`; tar entry `size` and zip `size()` headers are not consulted either.
- Proposed fix: pass a per-entry ceiling (e.g. 2 GiB) and a per-archive total (e.g. 1.5 × the catalog `part.size` × 8, or a flat 6 GiB) into `write_entry` via `reader.take(limit + 1)` and fail when `n > limit`; also refuse entries whose declared size exceeds the limit before opening them.
- Phase: 3

### RS-05  Oversize bodies are retried forever and `Content-Length` is never checked against the pinned size
- Severity: resource-exhaustion
- Location: src/resources/manager.rs:1168-1175 (verified `if have > part.size` / "is larger than the pinned"), 207-209 (`retryable`), 1096-1183 (`download_part`)
- Trigger: the pinned URL starts serving a different (larger) file, or a captive portal / proxy returns a 200 HTML page bigger than a small part (class `.yaml` files are 23-1193 bytes). The service (`retry: true`) deletes the `.partial`, waits 30 s..5 min and downloads up to `part.size` bytes again, forever; for `nvidia-cuda-libs` that is ~700 MB per attempt. A body that is *smaller* with a 200 is reported as "interrupted (will resume)" and the next attempt sends `Range`, which the same server answers with 200 again (truncate, repeat).
- Evidence:
  ```rust
  if have > part.size {
      drop(f);
      std::fs::remove_file(partial).ok();
      bail!("{} is larger than the pinned {} bytes; the file was deleted", ...)
  ```
  while `retryable()` treats only `HashMismatch` and `UrlRefused` as permanent.
- Proposed fix: compare `resp.content_length()` (and the `Content-Range` total) with `part.size - have` before streaming and fail permanently on mismatch (a new `SizeMismatch` error treated like `HashMismatch`); make the "larger than pinned" error permanent too; cap automatic attempts (e.g. 20) per job.
- Phase: 3

### RS-08  Child output lines are buffered without a length bound
- Severity: resource-exhaustion
- Location: src/resources/export.rs:677-695 (verified `fn spawn_reader` 677, `reader.split(b'\n')` 684)
- Trigger: uv and Ultralytics/tqdm progress bars rewrite one line with `\r` and emit `\n` rarely; a one-hour `uv pip install` or a chatty export can produce many MB between newlines, all held in one `Vec<u8>` per stream until the newline arrives (the `\r` split happens only afterwards). A misbehaving child (binary garbage, no newline) grows memory until the stage timeout.
- Evidence:
  ```rust
  for chunk in reader.split(b'\n') {
      let Ok(chunk) = chunk else { break };
      for part in chunk.split(|b| *b == b'\r') {
  ```
- Proposed fix: read with `BufRead::fill_buf`/`read_until` treating both `\n` and `\r` as delimiters at read time and cap a single segment (e.g. 64 KiB, truncate with "...").
- Phase: 3

### RS-12  Release-check JSON body is unbounded and cached
- Severity: resource-exhaustion
- Location: src/update.rs:30, 49 (verified `body: Option<String>`, `resp.json::<Release>()`)
- Trigger: a hostile or broken response from `api.github.com` (or a MITM on a box without proper CA roots) containing a multi-MB `body`: `resp.json` buffers it (bounded only by what 10 s can transfer), it is stored in `CACHE` for an hour and forwarded as `releaseNotes` to every `/status/updates` caller. Malformed JSON and odd `tag_name` values are handled (`success: false`, no panic); nothing is auto-applied.
- Evidence:
  ```rust
  resp.json::<Release>().await.context("parsing release JSON")
  ```
- Proposed fix: read with `resp.bytes()` after checking `content_length() <= 256 KiB` (or `take()` the stream), then `serde_json::from_slice`; truncate `releaseNotes` to a few KB. Verify (out of scope here) that the template renders `releaseNotes` escaped.
- Phase: 3

### SV-04  No request-head, idle or in-flight limits on the listener; each upload is fully buffered
- Severity: resource-exhaustion
- Location: src/server.rs:334-356 (`serve_on`), src/server.rs:190 (`DefaultBodyLimit`), src/server/detect.rs:52-55 (`field.bytes().await`) (verified)
- Trigger: (a) a LAN client opens N TCP connections and sends partial request heads (slowloris): hyper
  1.12's `header_read_timeout` default of 30 s only works when a `Timer` is installed, and
  `axum::serve` 0.8.9 installs none (checked in the vendored sources: no `timer(` in `axum/src/serve`),
  so the connections stay open indefinitely (runtime behaviour unverified on this machine);
  (b) N concurrent `POST /v1/vision/detection` with 32 MiB bodies: every body is read into memory
  (`field.bytes()`) before the queue check, so memory is N × 32 MiB with no cap on N; the worker queue
  bound only applies after the copy. (c) idle keep-alive connections are never closed by the server.
- Evidence: `let v4 = axum::serve(listeners.v4, service()).with_graceful_shutdown(...)` — no
  `tower::limit::ConcurrencyLimitLayer`, `TimeoutLayer`, `http1().header_read_timeout`, or
  `.timer(TokioTimer::new())`.
- Proposed fix: serve through `hyper_util::server::conn::auto::Builder` with
  `.timer(TokioTimer::new())`, `header_read_timeout(10 s)` and an idle timeout, add a
  `ConcurrencyLimitLayer`/`GlobalConcurrencyLimitLayer` (e.g. 2 × total queue capacity + 32) and a
  `TimeoutLayer` (request timeout + grace) on the detection routes; consider rejecting a detection
  before reading the body when the worker is already full (trade-off with the connection-reuse note
  in detect.rs:86-88: read-and-discard with a small cap instead).
- Phase: 3

### SV-08  Unauthenticated clients can fill the in-memory log ring with ~800 MiB via the unknown-model warning
- Severity: resource-exhaustion
- Location: src/server/detect.rs:106 (`warn!("request for unknown model '{model}'")`), src/logbuf.rs:103-121 (`push`: bounded by entry count only) (verified)
- Trigger: `POST /v1/vision/custom/<400 000 × "a">` (hyper accepts request heads up to
  `DEFAULT_MAX_BUFFER_SIZE` = 8192 + 4096×100 ≈ 408 KiB, `hyper-1.12.0/src/proto/h1/io.rs:23`), repeated
  2000 times from any LAN client. Each request logs the percent-decoded segment at `warn` (above the
  default `info` level, so it is always captured), and `LogBuffer::push` keeps the whole message:
  2000 × ~400 KiB ≈ 800 MiB resident until evicted; the same text also goes to the log file / event log
  (log injection: `%0A` in the path becomes a newline in the message). The `/logs.json` reply and the
  Logs page then ship up to 800 MiB per poll.
- Evidence: `warn!("request for unknown model '{model}'");` and `r.entries.push_back(LogEntry { ...
  message, })` with `capacity` counted in entries.
- Proposed fix: truncate and sanitise client-supplied strings before logging (e.g. first 128 chars,
  control characters escaped) and cap `LogEntry.message` length in `push` (e.g. 4 KiB, with a byte
  budget for the ring); rate-limit the unknown-model warning (once per name per minute, or `debug`).
- Phase: 3

### BE-03  `OrtRuntime::full_name` names the wrong GPU when CUDA ordinals differ from the platform order
- Severity: correctness
- Location: src/backend/ort.rs:571-583 (`full_name`), versus src/backend/select.rs:747-762 (`cuda_ordinals`)
- Trigger: two or more NVIDIA GPUs where the driver's ordinal order differs from the DXGI/DRM order (the very case `select::tests::cuda_ordinals_follow_the_driver` encodes: ordinal 0 = RTX 4090 at index 2, ordinal 1 = RTX 3060 at index 0). `ort:cuda:0` is compiled on the 4090 (`device_id` = ordinal) but `full_name` does `nvidia.sort_by_key(index); nvidia.get(0)` and reports "NVIDIA GeForce RTX 3060" in `executionProvider`, the status page and logs.
- Evidence: `let mut nvidia: Vec<_> = self.hw.gpus_of(GpuVendor::Nvidia).collect(); nvidia.sort_by_key(|g| g.index); nvidia.get(index as usize)` (ort.rs:574-579).
- Proposed fix: `select::cuda_ordinals(&self.hw).into_iter().find(|(o, _)| *o == index).map(|(_, g)| g.name.clone())`, and add a unit test with the same fixture as the select test.
- Phase: 4

### BE-05  Unchecked i64 arithmetic on values parsed from the model in the Split/Flatten emitters
- Severity: correctness
- Location: src/backend/onnx_rewrite.rs:917 (`let end = start + size;`), :889 (`(dim + k - 1) / k`), :969 (`lead.iter().product()`)
- Trigger: a crafted or corrupt opset<13 model with a `Split` whose `split` attribute holds values near `i64::MAX` (the `>= 0` check at 912 passes) reaches 917 under `RewriteOptions::COREML`. Debug/test builds panic (the parser is otherwise carefully non-panicking); the release profile (Cargo.toml:99-103, no `overflow-checks`) wraps and writes nonsensical Slice bounds into the cache file, which ORT then rejects at session creation (the plan falls back to CPU). 889/969 need `RewriteOptions::ALL` (tests only today).
- Evidence: `let mut start = 0i64; for (j, (size, out)) in sizes.iter().zip(&n.outputs).enumerate() { let end = start + size;` (onnx_rewrite.rs:915-917).
- Proposed fix: use `checked_add`/`checked_mul` and return `Err("split sizes overflow")` so the node is skipped with a reason; add a test with `split = [i64::MAX, 1]`.
- Phase: 1

### BE-07  Device index accepted up to `u32::MAX` and cast to `i32` for ORT `device_id`
- Severity: correctness
- Location: src/backend/spec.rs:276-281 (`parse_index` accepts any `u32`), src/backend/ort.rs:693 (`let device_id = index as i32;`)
- Trigger: config `"device": "ort:cuda:4294967295"` (or any N > 2147483647) parses fine and becomes `device_id = -1`; ORT rejects it with a provider error that names a negative id rather than "index out of range". `openvino:gpu.99999` similarly becomes `GPU.99999` and fails inside the plugin. Operator-controlled input, so an error not a crash.
- Evidence: `s.parse().ok()` on `u32` (spec.rs:280); `let device_id = index as i32;` (ort.rs:693).
- Proposed fix: cap the index in `parse_index` (e.g. `<= 255`, or `<= i32::MAX`), and in `plan`/`compile` report "no device with index N (have M)" using `HardwareInfo`/OpenVINO device lists.
- Phase: 4

### BE-09  `device_class` for a bare `openvino:gpu` may classify a different GPU than the one `load_model` picked
- Severity: correctness
- Location: src/backend/mod.rs:741-765 (`device_class`), src/backend/device.rs:421-437 (`prefer_intel_gpu`)
- Trigger: Windows machine with an NVIDIA card plus an Intel iGPU where OpenVINO's OpenCL plugin lists both (`GPU.0` = NVIDIA, `GPU.1` = Intel). `load_model` redirects bare `GPU` to `GPU.1` (Intel); `device_class` compares "GPU" with `GPU.0`/`GPU.1`, finds no match and uses `gpus.first()` = `GPU.0`'s `DEVICE_TYPE` (discrete) -> `DeviceClass::DiscreteGpu` -> THROUGHPUT-style tuning for an iGPU.
- Evidence: `.find(|d| d.name.eq_ignore_ascii_case(&name)).or_else(|| gpus.first())` (mod.rs:756-757).
- Proposed fix: resolve the class from the loaded `DeviceInfo::actual` (the compiled device name) instead of the requested spec, or apply the same `prefer_intel_gpu` resolution in `device_class`.
- Phase: 4

### BE-14  OpenVINO `submit` bumps the sequence number before `infer_async`; a synchronous start failure may leave a live completion
- Severity: correctness
- Location: src/backend/device.rs:981-986
- Trigger: `infer_async()` returns an error (e.g. GPU context lost). `seq` was already incremented and `busy` stays false. If the plugin still invokes the completion callback for the failed start (OpenVINO fires callbacks "also after a failure" per the comment at 601-602; whether that includes a synchronous start failure is unverified), the worker receives a `Completion { seq: new }` for a slot it never marked busy and must ignore it; `collect` on it would error "no inference in flight".
- Evidence: `let seq = s.ctx.seq.fetch_add(1, Ordering::AcqRel) + 1; s.request.infer_async().with_context(...)?; s.busy = true;` (device.rs:981-985).
- Proposed fix: either set `busy = true` before `infer_async` and reset it (after a `wait_done`) on error, or have the worker drop completions whose slot is not busy (document which side owns this).
- Phase: 4

### BM-02  `--min-confidence` is not validated; NaN or inf is persisted as `null` and makes benchmark.json unloadable
- Severity: correctness
- Location: src/benchmark/cli.rs:163, 318, 519
- Trigger: `blue-onyx-prism benchmark --all-devices --min-confidence nan` (Rust's f32 parser accepts `nan`/`inf`). Values < 0 or > 1 are also accepted.
- Evidence:
  ```rust
  pub min_confidence: Option<f32>,
  ...
  p.job.confidence_threshold = Some(c);
  ...
  confidence_threshold: args.min_confidence.unwrap_or(config.confidence_threshold),
  ```
  The threshold flows into `AccuracyReport.threshold` (f32, required) and `ThresholdSweep.configured`. serde_json writes non-finite floats as `null`, which does not deserialize back into `f32`. The file then fails to load (auto order is lost, and `--threshold-search` errors), and the next save drops all models (BM-01). With NaN, every production prediction is also filtered out (`conf >= NaN` is false). The config path validates thresholds (config.rs:612); the CLI bypasses that.
- Proposed fix: give `--min-confidence` a clap `value_parser` that requires a finite value in 0..=1, matching `validate_models`.
- Phase: 1 (crash safety)

### BM-03  `merge` drops all previous results on any HardwareSummary string difference, inconsistent with `same_gpus`
- Severity: correctness
- Location: src/benchmark/report.rs:908 (vs `same_gpus` at 593-602)
- Trigger: previous results were recorded with GPU strings that differ from the current ones only in the optional CUDA suffix, which `GpuAdapter`'s Display appends when `cuda` is `Some` (backend/detect.rs:179-185). Concrete cases: an upgrade from a version that did not print CUDA facts (the test at report.rs:1835 documents such files), or CUDA detection changing between runs (unverified which conditions flip it). A VRAM or CPU-name string change has the same effect. After that, benchmarking one model wipes every other model's results.
- Evidence:
  ```rust
  if prev.hardware != new.hardware || prev.runtimes != new.runtimes {
      return new;
  }
  ```
  `auto_order` deliberately accepts the same files as "same GPUs" (prefix match on `#idx name (`).
- Proposed fix: decide comparability in `merge` with `same_gpus`-style logic (OS, arch, GPU index and name) plus runtime versions, not full struct equality. Also tell the user when previous results are dropped.
- Phase: 4 (operability)

### BM-10  Pseudo ground truth is capped at 100 reference detections per image
- Severity: correctness
- Location: src/benchmark/mod.rs:1191-1192 (truncate to `MAX_STORED_PREDS`), 1674-1685 (pseudo GT built from `r.preds`)
- Trigger: an image where the reference model reports more than 100 detections at ≥ 0.3 (crowds, parking lots, 4 MP scenes). The lowest-confidence ones are dropped from the pseudo ground truth: first ignore regions, then real objects if more than 100 are ≥ 0.5. Every benchmarked model's correct detections of the dropped objects then count as false positives, and the relative precision and AP shift.
- Evidence:
  ```rust
  preds.sort_by(|a, b| b.confidence.total_cmp(&a.confidence));
  preds.truncate(MAX_STORED_PREDS);
  ```
  `pseudo_ground_truth` reads `r.preds` (the truncated list), not the untruncated `eval_preds`.
- Proposed fix: build the pseudo GT from the run's untruncated eval predictions filtered at `PSEUDO_IGNORE_CONFIDENCE`, or skip the truncation for the reference run.
- Phase: 4 (operability)

### BQ-06  Matching in crowd/ignore regions uses IoU, not pycocotools' intersection-over-detection
- Severity: correctness
- Location: src/benchmark/metrics.rs:337-339 (doc claim at 12-15)
- Trigger: a COCO `iscrowd` box covering a group of people (coco-cctv: 21 crowd regions across 22 `crowd`-tagged images): an individual person detection inside it has a low IoU with the large crowd box, so it counts as a false positive, where pycocotools computes `inter / area(det)` for crowd GT and ignores it. AP50, precision and the threshold picks on crowd images are understated relative to COCO numbers.
- Evidence: `.any(|g| g.ignore && g.label == p.label && iou(&p.bbox, &g.bbox) >= iou_thr);`
- Proposed fix: for ignore regions use `inter / area(pred)` (pycocotools' crowd rule), or carry a `crowd` flag on `GtBox` so COCO crowd boxes use that rule while pseudo-GT/"other" ignore boxes keep IoU; add a test against a pycocotools-computed value.
- Phase: 4 (operability)

### BQ-07  A pick that misses its precision/FPR target at every 0.01 step is marked `met: true` and is applied
- Severity: correctness
- Location: src/benchmark/threshold.rs:367-394 (`make(t, true, note, ...)`), src/benchmark/threshold.rs:815-819 (`apply_value`)
- Trigger: objective `precision:0.9` where precision >= 0.9 holds only in a range narrower than 0.01 (e.g. between confidences 0.512 and 0.518): the rounded threshold has precision < 0.9, a note is attached, but `met` stays true, so **Apply threshold** / `benchmark --apply` writes a value that violates the target the user asked for.
- Evidence: `None => { ... note = Some(format!("{name} {}% is met only between 0.01 steps ...")) }` followed by `return Some(make(t, true, note, ...))`; `.filter(|b| b.met && b.point.counts.tp > 0)`
- Proposed fix: set `met` to `objective.target_met(&bp.point_at(t)) != Some(false)` (or use the unrounded exact threshold for target objectives), so `apply_value` refuses a target-violating value.
- Phase: 4 (operability)

### BQ-08  A COCO JSON that matches no image silently turns the set into "annotated, zero objects"
- Severity: correctness
- Location: src/benchmark/images.rs:640-655 (also basename collapse at 619-624)
- Trigger: `instances*.json` auto-discovered (562-580) but for another split (e.g. `instances_train2017.json` beside val images), or `file_name`s that differ from disk names (case, extension): every image gets `gt = Some([])`, the set is reported as "ground truth (COCO JSON)", all detections become FPs and AP is None, with no warning. Two COCO entries `a/x.jpg` and `b/x.jpg` collapse to `x.jpg`; the first wins.
- Evidence: `img.gt = Some(parsed.iter().find(|(f, _)| *f == img.file).map(|(_, b)| b.clone()).unwrap_or_default());`
- Proposed fix: count matched images; bail (or warn into `Resolved::warnings`) when none or only a small share match; detect duplicate basenames and warn. Using a HashMap also removes the O(images x entries) scan.
- Phase: 4 (operability)

### BQ-09  YOLO label values are not validated
- Severity: correctness
- Location: src/benchmark/images.rs:721-750 (and 499 / 421 where an unreadable size becomes 0x0)
- Trigger: label lines with `nan`/`inf` (Rust parses both as f32), negative or > 1 values, YOLO-segmentation polygon lines (`cls x1 y1 x2 y2 ...` silently read as `cx cy w h`), or an image whose header could not be read (w=h=0 -> every box collapses to zero size). Such objects can never be matched, so they count as misses and lower recall/AP without any message.
- Evidence: `let (cx, cy, bw, bh) = (v[0] * w, v[1] * h, v[2] * w, v[3] * h);` with `v` from `s.parse::<f32>()`; `image::image_dimensions(&path).unwrap_or((0, 0))`
- Proposed fix: require finite values in [0, 1] (error with line number), reject lines with more than 5 fields (or detect polygons and take their bounding box), and fail/skip images with unknown size when YOLO labels are applied.
- Phase: 4 (operability)

### CO-04  The config file, CLI flags and on-disk edits are never validated, and only the `/config` form path validates
- Severity: correctness
- Location: src/config.rs:380-387 (`load`, serde only, verified); src/cli.rs:206-224 and 324-399 (merge without checks); src/config_store.rs:223-233 (disk reload adopted as is); src/runner.rs:224-234 (restart reload). `validate_config` is only called at src/config_form.rs:51.
- Trigger: `"confidence_threshold": 5` or `-1`, `"nms_iou": 2`, `--confidence-threshold 7`, `"port": 0` (binds a random port silently), a global `"device": "xpu"` (only fails per model at load), or a hand edit that disables every model. Each of these is loaded, written back, and (through `check_disk`) applied live, so Blue Iris silently gets no detections or every box. When duplicate names occur, the registry refuses the restart and the previous config is restored, but the file keeps the bad value.
- Evidence:
  ```
  let mut config: Config = serde_json::from_str(&text)
      .with_context(|| format!("parsing config {}", path.display()))?;
  config.normalize();
  Ok(config)
  ```
- Proposed fix: `Config::load` runs a `validate()` that range-checks every numeric field, names the key and the range (`confidence_threshold: must be 0..=1, got 5`), and is shared by the form, `merge_cli` (reject before write-back) and `sync_disk` (set `disk_error`, keep memory).
- Phase: 1

### CO-08  An external edit with the same length and mtime tick is missed, then overwritten by the next UI save
- Severity: correctness
- Location: src/config_store.rs:263 (verified) and :210-215
- Trigger: a filesystem with 1-2 s mtime resolution (FAT/exFAT, some SMB or Docker bind mounts, HFS+). An editor or `blue-onyx-prism add-model` changes `"port": 4000` to `4001` within the same second as the last write. `check_disk` and `sync_disk` return early on equal (mtime, len) and never hash, so `update()` then writes the in-memory config over the external edit. The test at line 405 sleeps 20 ms and relies on fine mtime.
- Evidence:
  ```
  (Some(d), Some((m, l))) if d.modified == m && d.len == l => return None,
  ```
- Proposed fix: in `update()` (rare, already under the write lock), always read and hash the file. Keep the stat-only shortcut only for the frequent `check_disk` poll.
- Phase: 4

### CO-09  `pending()` does not report a removed per-model `inflight` (fields omitted when `None`)
- Severity: correctness
- Location: src/config_merge.rs:531 (verified, iterates only `oc`); src/config.rs:141-142 (`skip_serializing_if = "Option::is_none"`)
- Trigger: the running model has `"inflight": 2`, and the user deletes the key in the models JSON. `model_object(current)` has no `inflight` key, so the loop never compares it, and the Config page shows no "needs restart" for that model when the reload did not happen (or for any future field that uses `skip_serializing_if`).
- Evidence:
  ```
  for (f, v) in &oc {
      ...
      if or.get(f) != Some(v) {
  ```
- Proposed fix: iterate the union of `or` and `oc` keys (as `model_reloads` effectively does by comparing whole maps).
- Phase: 4

### CO-10  The CLI resolves a relative `log_path` from the config file against the cwd; the service resolves it against the exe dir
- Severity: correctness
- Location: src/bin/blue_onyx_prism.rs:348 (verified) vs src/bin/blue_onyx_prism_service.rs:137 (`resolve_path`)
- Trigger: the config has `"log_path": "logs"`, and the CLI is started by systemd with `WorkingDirectory=/` or from another directory. Logs go to `<cwd>/logs` (or fail to create), contrary to README line 198 ("Relative paths in the file resolve against the executable directory").
- Evidence:
  ```
  let log = cli::init_logging(config.log_level, config.log_path.as_deref())?;
  ```
- Proposed fix: pass `config.log_path.as_deref().map(blue_onyx_prism::resolve_path)`, as the service does.
- Phase: 4

### CO-14  Test client: `--interval-ms` delays compound with `buffered`, and a large interval panics
- Severity: correctness
- Location: src/bin/test_blue_onyx_prism.rs:170 (verified) and :68
- Trigger: `--repeat 10 --interval-ms 100 --parallel 1`. Request i is created only after request i-1 finishes and then sleeps `i*100 ms`, so the total sleep is 4.5 s, not 0.9 s, and the reported throughput is wrong. A large `--interval-ms` with many repeats overflows `Duration * u32` and panics. `(i / parallel) as u32` truncates.
- Evidence:
  ```
  let delay = interval * (i / parallel) as u32;
  ```
- Proposed fix: compute an absolute start time (`wall + interval * k`) and `sleep_until` it, using `checked_mul` with an error.
- Phase: 5

### NR-07  workflow_dispatch can rewrite an already-published release
- Severity: correctness
- Location: .github/workflows/release.yml:13, :144
- Trigger: a manual run on `main` after `X.Y.Z` was published, while Cargo.toml still says `X.Y.Z`. `tag_name` resolves to `X.Y.Z`. softprops/action-gh-release then targets the existing release for that tag and uploads the archives built from the current `main` HEAD, not the tagged commit, replacing same-named assets. This behavior is from the action's documentation for v2 (`overwrite_files` default true) and is unverified for v3.0.3. The `.sha256` files users already downloaded would then mismatch.
- Evidence: `tag_name: ${{ github.ref_type == 'tag' && github.ref_name || env.VERSION }}`, `draft: true`. The workflow has no check that a release for that tag already exists.
- Proposed fix: On `workflow_dispatch`, fail if `gh release view "$VERSION"` succeeds, or use a distinct tag such as `X.Y.Z-dev.<run>`. Set `overwrite_files: false`.
- Phase: 4

### NR-08  Ultralytics exports are not checked against PyTorch, although the docs say every export is
- Severity: correctness
- Location: scripts/export_model.py:364-371 (README.md:315, CLAUDE.md:38)
- Trigger: any `model:yolo*` export (Config page or `fetch --resource model:yolo11s`). Only the output port shapes are checked. A numerically wrong ONNX passes and is installed, for example from an onnxslim regression or an opset-17 conversion issue in a new Ultralytics pin.
- Evidence: `onnx = Path(model.export(format="onnx", ...))` / `inputs, outputs = onnx_ports(onnx)` / `check_ports(...)` / `shutil.copy2(onnx, out["onnx"])`. `compare_with_onnxruntime` is called only at lines 458 (RF-DETR) and 664 (DEIMv2). CLAUDE.md:38 says "static-shape ONNX checked against PyTorch with ONNX Runtime".
- Proposed fix: After the Ultralytics ONNX export, run the PyTorch model (`model.model` in eval mode on `synthetic_image(imgsz)`) and `compare_with_onnxruntime` on the top-k scores, as the HF recipes do. Otherwise correct the README and CLAUDE.md wording.
- Phase: 4

### NR-09  README Docker instructions put models where the compose file never mounts them
- Severity: correctness
- Location: README.md:618-621, deploy/docker-compose.yml:23
- Trigger: a user follows the README: `mkdir -p models`, copies models into `deploy/models`, and edits `deploy/config` to name them. The container mounts the named volume `models`, so `/app/models/<file>` is missing and the model is `Failed` (with `auto_download: false`, nothing is fetched).
- Evidence: README:621 "Put models in `deploy/models` and a `blue_onyx_prism_config.json` in `deploy/config`."; compose:23 `- models:/app/models` (named volume, declared at :29-30).
- Proposed fix: Change the README to "add models through the named volume, or replace `models:/app/models` with `./models:/app/models`". Alternatively, switch the compose file to the bind mount and bake the default model elsewhere.
- Phase: 4

### NR-10  README says "nothing downloaded is executed"
- Severity: correctness
- Location: README.md:120
- Trigger: a reader relies on this for a security assessment. The service dlopens the downloaded OpenVINO, ONNX Runtime, DirectML and CUDA libraries into its own process, and the export feature runs downloaded uv/Python. README:325 itself calls export "the one feature that runs downloaded code".
- Evidence: line 120: "only whitelisted libraries are extracted; nothing downloaded is executed."
- Proposed fix: Reword to "downloaded libraries are SHA-256 pinned and loaded into the process; no downloaded program is run except by a model export you start".
- Phase: 4

### NR-11  Stale or incomplete README statements
- Severity: correctness
- Location: README.md:184, :188, :231, :283, :413-414, :651
- Trigger: users following these lines get behavior other than described.
  - :184 "macOS: CPU is used automatically": `auto` picks `ort:coreml` on Apple silicon.
  - :283 "`--device GPU`, default": the default is `auto`.
  - :188 / :231 family lists omit `detr` and `rfdetr`, although `ModelFamilyKind` has them and README:405 needs `--family rfdetr`.
  - :413-414 "**Save and restart** applies the selection ... switching is a click plus a restart": plain **Save** applies at once (tests/enabled_models_http.rs:331-339, tests/live_config.rs:177-197).
  - :651 the `threshold-search` objective list omits `youden` and `fpr:<x>`, which README:493-496 documents.
- Evidence: quoted lines above.
- Proposed fix: Update those lines. Also add the missing config keys `preprocess_threads`, `inflight`, `ort_sessions`, `hardware_profile` and model `inflight` (src/config.rs:142, :220-229) to the configuration table.
- Phase: 4

### PL-02  Non-finite model outputs reach the response as `(0,0,0,0)` boxes and `"confidence": null`
- Severity: correctness
- Location: src/model/mod.rs:504-515, src/model/yolo5.rs:62, src/api.rs:58
- Trigger: a model (fp16 overflow on GPU, a bad custom export) emits NaN coordinates or `inf` logits; yolo5 `conf = obj * best` with `inf` passes `meets`.
- Evidence: `let x1 = d.x1.clamp(0.0, w); ... if x2 <= x1 || y2 <= y1 { return None; }` (false for NaN) and `x_min: x1.round() as usize` (NaN -> 0); serde_json writes non-finite `f32` as `null`, which Blue Iris does not expect.
- Proposed fix: in `to_predictions` skip detections where any of `x1,y1,x2,y2,score` is not finite (or clamp score to 0..=1); add a synthetic-tensor test with NaN/inf rows for yolo26/yolo5.
- Phase: 1

### PL-05  `family: auto` misdetects a 1-class YOLOv5 export as YOLO26
- Severity: correctness
- Location: src/model/mod.rs:396-406
- Trigger: a single-class YOLOv5 model (`[1, 25200, 6]`, class YAML with one name) with the default `auto` family: `last == 6` is tested before the `5 + num_classes` rule, so rows `cx,cy,w,h,obj,cls0` are decoded as `x1,y1,x2,y2,conf,cls` -> wrong boxes, class ids 0/1, no NMS, silently.
- Evidence: `if last == 6 { return Ok(ModelFamilyKind::Yolo26); } if num_classes > 0 && prev == 4 + num_classes as i64 && last > prev { ... Yolo8 } if num_classes > 0 && last == 5 + num_classes as i64 { ... Yolo5 }`
- Proposed fix: evaluate the class-count rules first; treat `last == 6` as YOLO26 only when `num_classes != 1` (or when `prev <= 1000`, YOLO26 exports have N = 300); add a test case.
- Phase: 4

### PL-08  Class index maps with gaps shift every later label
- Severity: correctness
- Location: src/model/classes.rs:32-49
- Trigger: a `names:` map such as `{0: person, 1: bicycle, 3: motorcycle}` (any YAML keyed by category id, e.g. a COCO-91 style or a pruned Ultralytics `metadata.yaml`): the list becomes `[person, bicycle, motorcycle]`, so class id 2 reports `motorcycle` and id 3 reports `class_3`.
- Evidence: `pairs.sort_by_key(|(i, _)| *i); Ok(pairs.into_iter().map(|(_, n)| n).collect())` (and `map_form` asserts `10: fire hydrant` lands at index 3).
- Proposed fix: build the list up to the maximum key (cap it, e.g. 10000) filling gaps with `class_{i}`, or reject non-contiguous maps with a message; update the test.
- Phase: 4

### RS-09  `install_outputs` moves two files and writes the manifest non-atomically
- Severity: correctness
- Location: src/resources/export.rs:1236-1275 (verified `fn install_outputs` 1236, `std::fs::rename(src, &dst)` 1264)
- Trigger: the `.onnx` rename succeeds and the `.yaml` rename (or the manifest write) fails (disk full, AV lock on Windows): the models dir now has a new `<name>.onnx`, a stale or missing `<name>.yaml`, and no (or an old) manifest. `is_exported` then returns true ("exported (by hand)" in the UI), the staging dir is removed by the caller, and the model loads with the wrong class file.
- Evidence:
  ```rust
  for (src, file) in [(&onnx, onnx_name), (&yaml, yaml_name)] {
      ...
      std::fs::rename(src, &dst)...?;
  ```
- Proposed fix: rename the `.yaml` first, then the `.onnx` (the `.onnx` is what makes the export "present"), and write the manifest `.tmp` before the final rename so a failure leaves either nothing new or a complete set; on error roll the `.yaml` back.
- Phase: 4

### RS-15  Range resume carries no validator; a changed upstream file becomes a permanent, non-retried failure
- Severity: correctness
- Location: src/resources/manager.rs:1115 (verified `RANGE, format!("bytes={have}-")`), 1082-1090 (`verify` → `HashMismatch`), 207-209
- Trigger: a download is interrupted at N bytes; before the retry the server's object changes (same URL, e.g. a re-uploaded release asset – the pin would then be stale anyway) or a proxy serves a different cached variant. The resumed request stitches two files; the size matches, the hash does not; the `.partial` is deleted and, because `HashMismatch` is permanent, nothing retries from scratch even though a clean download might succeed.
- Evidence:
  ```rust
  if have > 0 {
      req = req.header(reqwest::header::RANGE, format!("bytes={have}-"));
  }
  ```
- Proposed fix: store the first response's `ETag`/`Last-Modified` next to the `.partial` (e.g. `<file>.partial.meta`) and send `If-Range`; on a hash mismatch after a *resumed* download, delete and retry once from zero before declaring the permanent failure.
- Phase: 4

### SV-13  A detection request with a non-multipart content type is answered without draining the body
- Severity: correctness (low)
- Location: src/server/detect.rs:72-79 (`parse_detection_body`), 86-88 (comment) (verified)
- Trigger: a client posts `Content-Type: image/jpeg` (raw image) with a pooled keep-alive connection:
  the `Multipart` extractor is rejected before the body is read, the handler answers at once, hyper
  closes the connection after the response and the client's next request on that connection fails
  (the exact failure mode the comment at lines 86-88 and the test at server.rs:441-474 try to avoid).
- Evidence: `Err(e) => Err(format!("Expected a multipart/form-data body: {e}"))` returns without
  touching the request body.
- Proposed fix: take `Request` (or `Bytes` via `axum::body::to_bytes` with the limit) when the
  extractor fails and discard it, or document that non-multipart posts close the connection.
- Phase: 4

### SV-14  Models-card per-model threshold accepts any finite number (e.g. 1.5 or -1)
- Severity: correctness
- Location: src/server/live.rs:441-466 (`models_card_edit`) (verified)
- Trigger: `POST /config/models` with `enabled=ipcam-general&threshold.ipcam-general=1.5&action=save`
  (the browser input has `max="1"`, but curl/`live.js` with a hand-typed value does not enforce it):
  `v.parse::<f32>()` succeeds, `validate_config` (config.rs:623-635) checks only model names and the
  enabled set, and the worker then filters every detection out. The benchmark path rejects the same
  value (`apply_model_thresholds`, tested at benchmark.rs:1610-1621 `threshold=1.5` -> 400).
- Evidence: `Some(v.parse::<f32>().map_err(|_| anyhow::anyhow!("threshold: '{v}' is not a number"))?)`
  then `m.confidence_threshold = t;` with no range check.
- Proposed fix: route the Models card through `config_form::apply_model_thresholds` (one validator),
  or add `(0.0..=1.0).contains(&t)` here and a global check in `validate_config`.
- Phase: 4

### BE-08  Config `openvino_dir` silently ignored when `OPENVINO_INSTALL_DIR` is set
- Severity: operability
- Location: src/backend/libs.rs:85-99
- Trigger: a machine where an Intel OpenVINO install's `setupvars` exported `OPENVINO_INSTALL_DIR`, and the operator sets `openvino_dir` in the config (or a bundled `./openvino` exists). The env var wins and the only trace is a `tracing::debug!` note, surfaced in `diagnostics()` only when `Core::new` fails. A version mismatch then shows up as odd plugin errors.
- Evidence: `note(format!("configured openvino_dir {} ignored because {ENV_INSTALL_DIR} is set", ...))` (libs.rs:94-96) where `note` logs at debug level (libs.rs:67).
- Proposed fix: log that specific note at `warn!`, and show it on the status page's runtime section.
- Phase: 4

### BE-10  A dead ORT session thread disables its slot for good; `Drop` can block on a hung session
- Severity: operability
- Location: src/backend/ort.rs:1296-1304 (`submit` -> "session thread exited"), :1323-1328 (`collect`), :1379-1392 (`Drop` joins)
- Trigger: a panic inside `run_session` (e.g. an `ort` assertion) or an ORT abort on one session thread. Afterwards every `submit` on that slot fails with "session thread N exited" until the model is reloaded; with `ort_sessions = 1` the model is effectively down while reported as Ready. Separately, a session wedged in `Session::run` (driver hang) makes `abort`/`Drop` wait forever, blocking worker shutdown and config reloads that drop the backend.
- Evidence: `.map_err(|_| anyhow::anyhow!("{where_}: session thread {slot} exited"))?` (ort.rs:1303, 1327); `if let Some(t) = s.thread.take() && t.join().is_err()` (ort.rs:1387-1388).
- Proposed fix: surface a distinct error kind (or a `Backend::is_healthy()`) so the worker/registry reloads the model after a thread death; document the hang limitation and consider a bounded join (detach after a timeout, logging) in `Drop`.
- Phase: 4

### BE-13  `openvino:gpu.N` plan only checks that some GPU exists
- Severity: operability
- Location: src/backend/plan.rs:150-169
- Trigger: config `openvino:gpu.3` on a machine listing only `GPU.0`: the plan tries `GPU.3` (plugin error "device not found", logged as a compile failure) and then falls back to CPU, instead of the clearer "requested device GPU.3 not available (devices: [...])" note the no-GPU branch produces.
- Evidence: `if available_ov_devices.iter().any(|d| d.starts_with("GPU")) { vec![Candidate::new(device, false), fallback(cpu)] }` (plan.rs:153-154).
- Proposed fix: when `device.index` is `Some(n)`, require `GPU.n` (or bare `GPU` when n == 0 and only `GPU` is listed) in `available_ov_devices`, else take the note branch.
- Phase: 4

### BE-15  Windows `PATH` mutation is inherited by child processes
- Severity: operability
- Location: src/backend/libs.rs:288-312 (`prepend_to_path`), called at :141 (OpenVINO `bin` + TBB) and :764 (`cuda-libs`)
- Trigger: on Windows the service prepends `openvino/runtime/bin/intel64/Release`, `runtime/3rdparty/tbb/bin` and `onnxruntime/cuda-libs` to the process `PATH`. The on-demand model export (`resources/export.rs`, not in this review) later spawns `uv`/Python with that environment, so Python's own `openvino`/`tbb12.dll`/CUDA wheels can resolve to the service's copies first (version skew, "DLL load failed" in the export).
- Evidence: `unsafe { std::env::set_var("PATH", joined) };` (libs.rs:308).
- Proposed fix: snapshot the original `PATH` before prepending and pass it explicitly to child processes (`Command::env("PATH", original)`), or use `AddDllDirectory` instead of `PATH` for the service's own DLL resolution.
- Phase: 4

### BM-05  A cancel issued right after start can be lost (flag reset after the state lock is released)
- Severity: operability
- Location: src/benchmark/service.rs:351-369
- Trigger: `POST` start, then cancel arrives between the end of the locked block (state = Running) and `self.cancel.store(false)` at line 369. `cancel()` sees `running == true`, stores `true` and returns true to the UI. `start_job` then stores `false`, and the run continues to completion.
- Evidence:
  ```rust
  g.state = RunState::Running; ... g.run_id
  };
  self.cancel.store(false, Ordering::SeqCst);
  ```
- Proposed fix: reset the cancel flag inside the locked block before setting `state = Running` (and have `cancel()` store while holding the lock), so set-and-reset are ordered by `Inner`.
- Phase: 4 (operability)

### BM-06  The dataset download wait has no deadline, and model compile is not cancellable
- Severity: operability
- Location: src/benchmark/service.rs:529-568 (loop), src/benchmark/mod.rs:846-849 (compile under lock, no cancel check)
- Trigger: `manager.status(key)` keeps returning `None` or a queued state. Possible causes are a key mismatch between `Job::into_dir(r, dir).key()` and what `actions::download` queued, or the download thread stalled behind another large download (unverified which happens in practice). The run then shows "downloading datasets … (0%)" forever until the user cancels. Separately, a cancel during a long GPU compile waits for the compile to finish.
- Evidence:
  ```rust
  loop {
      if h.is_cancelled() { return Err(super::Cancelled.into()); }
      ...
      _ => pending.push(Some(id.clone())),
  ```
- Proposed fix: add an overall deadline, or a "no progress for N minutes" timeout that turns the remaining sets into warnings. Treat `None` status (unknown key) as a failure after a short grace period. Document that cancel takes effect after the current compile.
- Phase: 4 (operability)

### BQ-10  Dataset resolution: one bad folder fails the run, id collisions silently drop a set
- Severity: operability
- Location: src/benchmark/images.rs:857-858, 891-893
- Trigger: (a) a `dir:` dataset that exists but has no images (or a malformed COCO/YOLO file) aborts the whole `resolve` with `?`, while a missing built-in set only warns; (b) a folder dataset whose `name` equals another set's id (e.g. `name: "sample"` or two folders with the same name) is skipped with no warning.
- Evidence: `load_dir(&dir, gt.as_deref(), Some(&id)).with_context(...)?` / `if out.sets.iter().any(|s| s.id == set.id) { continue; }`
- Proposed fix: push a warning for the duplicate-id skip; decide and document whether bad folders are fatal (validation at config save time would surface it earlier).
- Phase: 4 (operability)

### CO-05  Windows service first start: default config has no models, so no HTTP UI and a 5-second retry loop forever
- Severity: operability
- Location: src/cli.rs:403-416 (verified, creates `Config { log_level: Debug, ..Default }` with `models: []`); src/runner.rs:36-44 and :351 (`ensure_models` before `bind` at :355); src/bin/blue_onyx_prism_service.rs:315-338
- Trigger: a fresh install with `scripts/install_service.ps1`, which seeds no config (grep shows no JSON write), then start. `run_server_on` fails with "no models configured" before binding, the service reports Running, and `http://127.0.0.1:32168/` (which the script tells the user to open, install_service.ps1:90) refuses connections. The README's "First run" assumes "a config that names a model".
- Evidence:
  ```
  let config = Config {
      log_level: LogLevel::Debug,
      ..Default::default()
  };
  ```
- Proposed fix: have `for_service` (and the CLI with no config) seed the default catalog model (`models/IPcam-general.onnx`, which `auto_download` fetches), or bind and serve the UI with an empty registry so models can be added from the Config page.
- Phase: 4

### CO-11  Config handling before logging is set up loses its messages, and a bad `log_path` aborts the CLI server
- Severity: operability
- Location: src/bin/blue_onyx_prism.rs:346-348 (verified order); src/config.rs:23-28 (rename info and warn); src/cli.rs:220-222 (write-back warn); src/cli.rs:499-500
- Trigger:
  - Starting with only `blue_onyx_openvino_config.json` present: the rename happens silently.
  - A read-only install dir: "could not rename" and "could not save merged config" are dropped, because no subscriber is installed yet.
  - An unwritable `log_path`: the CLI exits with an error, while the service falls back to stdout and the event log.
- Evidence:
  ```
  let (config, config_path) = cli::resolve_config(&cli)?;
  // Initialized once; ...
  let log = cli::init_logging(config.log_level, config.log_path.as_deref())?;
  ```
- Proposed fix: collect those messages and log them after `init_logging`, or `eprintln!` them. In the CLI, fall back to stdout logging with a warning when the log dir cannot be created.
- Phase: 4

### CO-12  Shutdown cannot be forced: a second signal is ignored and the drains have no upper bound
- Severity: operability
- Location: src/bin/blue_onyx_prism.rs:372-391 (verified, one-shot); src/runner.rs:327-330 (`block_on(server task)`, no timeout); src/bin/blue_onyx_prism_service.rs:251-258 (stop loop with no deadline)
- Trigger: Ctrl-C while a model is compiling on the GPU (minutes) or while a slow client holds a request. The process waits for graceful completion, and a second Ctrl-C does nothing. In the service, the SCM sees StopPending checkpoints that keep advancing, so the stop takes as long as the slowest worker. When the `tokio::runtime::Runtime` is dropped at the end of `main`, it also waits for any `spawn_blocking` task (unverified for long benchmark or export tasks).
- Evidence:
  ```
  let served = server_task.map(|t| {
      rt.block_on(t)
  ```
- Proposed fix:
  - Exit on a second signal (`std::process::exit(130)`).
  - Bound `finish` with `tokio::time::timeout`.
  - Use `rt.shutdown_timeout(...)`.
  - In the service, stop waiting after about 60 s and report Stopped with a service-specific code.
- Phase: 4

### CO-13  Windows memory detection spawns `powershell` by search path with no timeout
- Severity: operability
- Location: src/system_info.rs:83-91 (verified)
- Trigger: every process start, through `Profile::detect` (OnceLock) and the startup log line, which calls it a second time. A slow or hung WMI/CIM provider, constrained-language policy or a missing `powershell` stalls startup or returns `None`, and `None` silently changes the hardware profile class. The name is resolved through the search path rather than `%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe`.
- Evidence:
  ```
  let out = std::process::Command::new("powershell")
      .args([ "-NoProfile", "-Command",
          "(Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory",
  ```
- Proposed fix: call `GlobalMemoryStatusEx` through the `windows` crate (already a dependency for DXGI), or at least use the absolute path plus a timeout, and cache the value in `system_info`.
- Phase: 4

### NR-12  The release checklist's CUDA check runs on a GPU the pinned Windows CUDA build cannot use
- Severity: operability
- Location: docs/RELEASE_CHECKLIST.md:13-20 (from docs/REVIEW_PLAN.md:369)
- Trigger: on the "GTX 1070 Ti box" (compute capability 6.1, PowerShell, so Windows), `benchmark --all-devices` never lists `ort:cuda` or `ort:tensorrt` as runnable. The pinned Windows CUDA build has kernels only for sm_75+. The step passes with DirectML/OpenVINO results, and CUDA/TensorRT ship without ever being exercised.
- Evidence: src/resources/catalog.rs:95-97: "SASS for sm_75, sm_86 and sm_89 ... So 7.5 is the minimum, and 6.x (Pascal) ... fail". tests/select.rs `pascal_dgpu_uses_directml_before_the_igpu` asserts CUDA is not runnable on that card. RELEASE_CHECKLIST:13 "On the GTX 1070 Ti box, with the release build and `setup-onnxruntime --flavor cuda` installed".
- Proposed fix: Run the CUDA/TensorRT step on a Turing-or-newer GPU, or run it on Linux, where the CUDA archs include 60. Make the step fail when the report has no `ort:cuda` row.
- Phase: 4

### NR-13  No release gate for the primary target's Intel iGPU path
- Severity: operability
- Location: docs/RELEASE_CHECKLIST.md:6-21, .github/workflows/ci.yml:47-56
- Trigger: a regression in `openvino:gpu` (f16 precision, the GPU plugin, cache, UHD 630 on OpenVINO 2026.4) on Windows + i5-8500, the primary target. Hosted runners have no Intel GPU, so tests/backend_cpu.rs exercises only the fallback branch (`if !has_gpu`). The checklist has no iGPU step.
- Evidence: the checklist's only manual hardware step is the NVIDIA one. ci.yml smoke uses `--force-cpu`.
- Proposed fix: Add a checklist step: run `benchmark --all-devices --report igpu.html` on the i5-8500 box, confirm `openvino:gpu` agrees with the CPU, and attach the report. Optionally add a self-hosted runner.
- Phase: 4

### NR-14  CI's "real model" tests skip every non-IPcam family and the lazy and RT-DETR/CoreML paths
- Severity: operability
- Location: .github/workflows/ci.yml:43-44, :59-66; tests/families_http.rs:21-25; tests/multi_model_http.rs:95-132; tests/backend_ort.rs:144-147, :176-205; tests/enabled_models_http.rs:240-243
- Trigger: CI downloads only `ipcam-bird` and `IPcam-general`, so `optional_model(...)` returns None. In CI:
  - YOLO26 (explicit and `auto` family) and RT-DETR are never run end to end.
  - The lazy-model HTTP flow is never run.
  - The RT-DETR i64 input on ORT is never run.
  - The RT-DETR-on-CoreML pinned-batch path is never run. That path guards against an MPSGraph process abort (CLAUDE.md).
  - The model-switch test uses an IPcam copy.
- Evidence: families_http.rs:21-25 candidates `yolo26s.xml`, `yolo26n.xml`, `rt-detrv2-s.onnx`, all through `common::optional_model`. ci.yml downloads only `--name ipcam-bird` (:44) and `--name IPcam-general` (:60).
- Proposed fix: Add `download-models --name rt-detrv2-s --dir target/release/models` (it is in the catalog, catalog.rs:1085) to CI. In multi_model_http, make the lazy model an IPcam copy when `yolo26n.xml` is absent, so the lazy flow always runs. YOLO26 needs an export (AGPL weights), so leave it to the release checklist and say so.
- Phase: 4

### NR-15  smoke.sh accepts a detection with zero predictions; provider pattern is an unescaped regex
- Severity: operability
- Location: scripts/ci/smoke.sh:53-58, :64-66
- Trigger: a provider path that returns `success:true` with an empty `predictions` array still passes the smoke test. Examples: a CoreML/DirectML numeric failure, or wrong preprocessing on a new runtime. For the Windows DirectML (WARP) run this smoke test is the only end-to-end coverage. Separately, an `EXPECTED` containing regex metacharacters (e.g. `OpenVINO GPU (`) breaks `grep -E`.
- Evidence: `echo "$OUT" | grep -q '"success":true'` and `grep -Eq "\"executionProvider\":\"[^\"]*$EXPECTED"`. No check on `"count"` or `predictions`.
- Proposed fix: Also require `"count":[1-9]`, or check for a `"label":"dog"`-style expected label, given the known test image and `min_confidence=0.2`. Use `grep -F` on the provider value extracted with a fixed pattern, and fail fast if `kill -0 $PID` shows the server died.
- Phase: 4

### NR-16  install_service.ps1 ignores sc.exe failures and passes the account password on the command line
- Severity: operability
- Location: scripts/install_service.ps1:61-63, :71, :73, :75-77
- Trigger:
  - A rerun while services.msc or Event Viewer holds a handle: `sc.exe delete` only marks the service for deletion, `sc.exe create` fails with 1072, and the script prints "Registering ..." and continues. Start-Service then fails or starts the old registration.
  - With `-Account`, the plain-text password is visible in the `sc.exe` process command line to other local users (WMI/Task Manager).
- Evidence: `sc.exe delete $ServiceName | Out-Null` and `sc.exe create ... obj= $Account password= $pw | Out-Null`. There is no `$LASTEXITCODE` check. `$ErrorActionPreference="Stop"` does not apply to native exit codes in PowerShell 5.1.
- Proposed fix: Check `$LASTEXITCODE` after each `sc.exe` and throw. After delete, poll `Get-Service` until it is gone (timeout). Use `New-Service -Credential $cred` instead of `sc.exe ... password=`.
- Phase: 4

### NR-17  Release archives and the binary ship third-party code without its license notices
- Severity: operability
- Location: .github/workflows/release.yml:93-98, LICENSE:25-49, src/resources/extract.rs:121 (only the DEIMv2 layout keeps `LICENSE`)
- Trigger: every release archive bundles `openvino/` (Apache-2.0, which requires the license text and NOTICE), `onnxruntime/` (MIT + ThirdPartyNotices) and, on Windows, `DirectML.dll` (Microsoft DirectML license). The binary embeds Roboto Mono (Apache-2.0). The whitelist extraction drops the license files, and LICENSE lists only the benchmark datasets.
- Evidence: release.yml:95 `setup-openvino --dest "$STAGE/openvino"`, :98 `setup-onnxruntime ... --dir "$STAGE/onnxruntime"`, :93 `cp README.md LICENSE "$STAGE/"`. LICENSE's third-party section covers only `assets/bench/*.json`. src/image.rs:4 names Roboto Mono (Apache-2.0).
- Proposed fix: Add a `THIRD_PARTY_LICENSES` file (OpenVINO, ONNX Runtime + ThirdPartyNotices, DirectML, Roboto Mono, Rust crates via `cargo about`/`cargo deny list`). Copy it into the archive, and extend the whitelist to keep each runtime's license file.
- Phase: 4

### NR-18  Docker image has no HEALTHCHECK; builder image tag floats
- Severity: operability
- Location: Dockerfile:9, :20, :36-38
- Trigger: a container whose server is stuck or never binds is reported healthy by Docker/compose (`restart: unless-stopped` only reacts to exit). Separately, a rebuild silently changes the builder image (`rust:1-bookworm`), and its glibc assumption (comment at :14) can break if the tag moves to a newer Debian.
- Evidence: `curl` is installed at :20 but never used, and there is no `HEALTHCHECK` line. `FROM rust:1-bookworm AS builder`.
- Proposed fix: Add `HEALTHCHECK --interval=30s --timeout=5s --start-period=120s CMD curl -fsS http://127.0.0.1:32168/stats.json >/dev/null || exit 1` (a non-admin route), and pin `rust:1.99.0-bookworm@sha256:...`.
- Phase: 4

### NR-19  Bind-mounted config and cache directories may not be writable by the image's `openvino` user (unverified)
- Severity: operability
- Location: Dockerfile:34-38, deploy/docker-compose.yml:24-27
- Trigger: `docker compose up` without the `mkdir` from README:617, so Docker creates `./cache` and `./config` as root. Or the host user's UID differs from the image's `openvino` UID. Then the entrypoint's `cp /app/docker-config.json /app/config/...` fails, `;` continues, and the server starts with a non-existent `--config` path, which it cannot write back.
- Evidence: `ENTRYPOINT ["/bin/sh","-c","[ -f /app/config/blue_onyx_prism_config.json ] || cp ... ; exec /app/blue-onyx-prism --config /app/config/blue_onyx_prism_config.json ..."]`; `USER openvino`. The UID of `openvino` in `openvino/ubuntu24_runtime:2026.4.0` was not checked here.
- Proposed fix: Make the entrypoint fail loudly (`cp ... || { echo "config dir not writable by $(id -u)"; exit 1; }`), and document `user:`/UID or `chown` in the README.
- Phase: 4

### NR-20  The weekly URL check reports nowhere a human reliably sees; jobs have no timeouts
- Severity: operability
- Location: .github/workflows/check-urls.yml:8-10, :25; .github/workflows/ci.yml:12-18
- Trigger:
  - A moved or deleted pinned URL makes the scheduled run go red. Nobody is notified except GitHub's default email to the cron's last editor. After 60 days without repository activity GitHub disables the schedule silently.
  - `cargo run` without `--locked` may resolve different versions than the lockfile.
  - A hung ignored test (e.g. a provider deadlock) holds a runner for the default 6 h.
- Evidence: check-urls:25 `run: cargo run --quiet -- list-resources --check-urls` (no `--locked`, no failure step). Neither workflow has `timeout-minutes`.
- Proposed fix: Add `--locked`. Add an `if: failure()` step that opens or updates an issue (`gh issue create`, needs `issues: write` on that job only). Set `timeout-minutes` (e.g. 60 for CI, 20 for check-urls).
- Phase: 4

### NR-21  Install/uninstall scripts are incomplete for upgrade and removal
- Severity: operability
- Location: scripts/install_systemd.sh:27, scripts/install_launchd.sh (no counterpart uninstall), scripts/install_service.ps1:47, scripts/uninstall_service.ps1:1-38
- Trigger:
  - Re-running install_systemd.sh after copying a new release: `systemctl enable --now` does not restart a running unit, so the old binary keeps running.
  - There is no Linux or macOS uninstall script.
  - Uninstall on Windows leaves the machine-wide `ServicesPipeTimeout=600000` in place, which affects every service on the box.
  - install_systemd.sh and install_launchd.sh are mode 100644 in git (`git ls-files -s`), so `sudo scripts/install_systemd.sh` from a checkout fails.
- Evidence: `systemctl enable --now blue-onyx-prism`; `Set-ItemProperty ... ServicesPipeTimeout -Value 600000` with no revert in uninstall_service.ps1.
- Proposed fix: Use `systemctl enable blue-onyx-prism && systemctl restart blue-onyx-prism`. Add `uninstall_systemd.sh` / `uninstall_launchd.sh`. Record the previous `ServicesPipeTimeout` and restore it on uninstall, or document why it is left. `git update-index --chmod=+x` both scripts.
- Phase: 4

### PL-06  A hung inference stalls the model forever: `unstick` converts a lost completion into a blocking `collect`
- Severity: operability
- Location: src/worker.rs:1018-1030, src/worker.rs:1248
- Trigger: a backend inference that never completes (driver hang, GPU reset); after `request_timeout` the tick marks it done and `finish` calls `eng.backend.collect(slot)` on the worker thread; every later request is accepted (`has_room` true while `pending < slots + 4`) and times out at the server; the model stays `Ready` with no recovery.
- Evidence: `"no completion after the request timeout; waiting for the inference"` then `f.done = Some(Instant::now());` and `let result = eng.backend.collect(slot)`
- Proposed fix: after N consecutive timeouts (or a second request timeout without `collect` returning) mark the model `Failed("inference stuck on <device>")` so the server rejects fast and the operator/UI can reload; document the deliberate blocking wait.
- Phase: 4

### PL-11  A reload can block the config-save path behind a compiling model
- Severity: operability
- Location: src/registry/plan.rs:48-51, src/registry.rs:627-631
- Trigger: a config save that reloads model B while model A is compiling (runtimes mutex held for the compile, worker.rs:1166) and B's plan has only runtime downloads pending with its file present: `first_runnable_wait` takes `lock_recover` and the caller (holding the `reload` mutex) waits for the whole compile; `ModelRegistry::selection` has the same blocking lock (no server caller found, so dormant).
- Evidence: `fn first_runnable_wait(...) { let rt = self.0.lock_recover(); rt.first_runnable(spec, model) }`
- Proposed fix: use `try_lock` with a short bounded wait (e.g. 2 s) and fall back to "wait for downloads" when busy; drop or `try_lock` `ModelRegistry::selection`.
- Phase: 4

### RS-04  A dead download worker leaves jobs "queued" forever: `install_all`, the exporter and blocked models hang
- Severity: operability
- Location: src/resources/manager.rs:642-658 (worker early returns), 488 (`let _ = self.tx.send`), 523-545 (`wait` with `None` loops), src/resources/export.rs:748-760 (`ManagerFetcher::fetch` polls), src/resources/provision.rs:330-338 (`Wait::poll` defaults to `Queued`)
- Trigger: `tokio::runtime::Builder::build()` or `reqwest::Client::build()` fails in the worker (no TLS roots, fd/thread limits), or the worker thread panics for any reason. `enqueue` still records `State::Queued` and `send` errors are ignored; nothing ever flips the status.
- Evidence:
  ```rust
  Err(e) => {
      tracing::error!("download manager: creating tokio runtime: {e}");
      return;
  }
  ...
  let _ = self.tx.send(Msg::Job(Box::new(job)));
  ```
- Proposed fix: have the worker mark every job it will never run as `Failed { retry_at: None }` before returning (and keep a `dead: AtomicBool`); make `enqueue` set `Failed` immediately when `tx.send` fails or `dead` is set; give `install_all`/`wait(None)` an overall cap or a "worker gone" check.
- Phase: 4

### RS-06  Service downloads retry without bound; a blocked model never reaches a terminal state
- Severity: operability
- Location: src/resources/provision.rs:128 (`mo.retry = true`), src/resources/manager.rs:746 (`retries.push`), 50-56 (`backoff` caps at 300 s but not attempts)
- Trigger: a URL that is unreachable for good (offline appliance, firewall, removed release asset giving 404) with `auto_download: true`: the job cycles Failed(retry in 300 s) → Queued → Failed forever, and the model stays `Initializing` with "download of ... failed ...; retrying in N s" indefinitely (`Wait::poll` returns `Failed` only for `retry_at: None`).
- Evidence:
  ```rust
  let retry = shared.opts.retry && retryable(&e);
  ...
  if retry { retries.push((Instant::now() + delay, job, attempts)); }
  ```
- Proposed fix: add `max_attempts` to `ManagerOptions` (service: e.g. 12 ≈ 1 h) after which the job becomes `Failed { retry_at: None }` with the manual `fetch` command; keep a UI Download button to re-queue. Document the bound in the module doc.
- Phase: 4

### RS-10  Service stop can block on an uninterruptible verify/extract of a large resource
- Severity: operability
- Location: src/resources/manager.rs:551-557 (`shutdown` joins), 1070-1092 (`verify` hashes the whole file), 846-900 (extraction under `run_job`)
- Trigger: stop the service while `nvidia-cuda-libs` (700 MB wheels) is in `Verifying`/`Extracting`: `stop` is polled only per HTTP chunk and in the lock loop; `Drop for Manager` joins the thread, so shutdown waits for the hash + extraction of the current part (tens of seconds on slow disks). `extract_part` cannot be cancelled at all.
- Evidence:
  ```rust
  pub fn shutdown(&self) {
      self.shared.stop.store(true, Ordering::Relaxed);
      let _ = self.tx.send(Msg::Stop);
      ... let _ = h.join();
  ```
- Proposed fix: check `stop` between parts and inside `sha256_file`/`extract_*` loops (pass `&AtomicBool`), aborting with "cancelled" (staging is removed on error anyway); or detach the thread on shutdown after a bounded join (e.g. 5 s) and log.
- Phase: 4

### RS-11  CLI export has no cancellation and waits on the export lock forever
- Severity: operability
- Location: src/resources/export.rs:1677-1683 (`export_now`, `let cancel = AtomicBool::new(false)`), 370-400 (`ExportLock::acquire` loop)
- Trigger: `blue-onyx-prism fetch --resource model:yolo26s` while the service is exporting: the CLI prints "another process is exporting" once and polls forever (no `--wait` bound). Ctrl-C terminates the CLI but, together with RS-02, leaves `uv`/`python` running.
- Evidence:
  ```rust
  let cancel = AtomicBool::new(false);
  ```
  and the lock loop has no deadline besides `cancel`.
- Proposed fix: wire a Ctrl-C handler (`tokio::signal` or `ctrlc`) to the cancel flag in the CLI path and give the lock wait a bound (e.g. 10 min) with a clear message.
- Phase: 4

### RS-14  `/v1/resources` rows do heavy filesystem and hardware probing per request
- Severity: operability
- Location: src/resources/status.rs:504-690 (`rows` 504, `export_row` 679, `toolchain_state(&paths, platform)` 689), src/resources/resolve.rs:1030-1086 (`detect_installed`), 389
- Trigger: the Config page polls `/v1/resources` while a download runs: each call runs `detect_installed` (`hardware()`, `system_cuda_present()` which loads CUDA libraries by name on NVIDIA boxes, several manifest reads), `resolve::needed`, `local_models` (directory scan) and, for each of ~40 export rows, `toolchain_state` (14 `exists()` calls) plus `export::manifest`. Hundreds of syscalls per poll; harmless on SSDs, noticeable on the Windows service with AV on-access scanning.
- Evidence:
  ```rust
  let paths = export::Paths::for_config(config);
  let toolchain = export::toolchain_state(&paths, platform);
  ```
  inside `export_row`, called per resource.
- Proposed fix: compute `toolchain_state`, `Platform::current()` and `unsupported_reason` once in `rows()` and pass them down; cache `detect_installed` for ~1 s or reuse the generation's `Provision.resolution` when nothing changed. Confirm the handler runs on `server::blocking` (not verified here).
- Phase: 4

### SV-06  Benchmark API handlers do blocking file I/O, multi-MB JSON parsing, config writes and CPU work on the 2-thread async runtime
- Severity: operability
- Location: src/server/benchmark.rs:570-614 (`start`), 618-660 (`settings`), 675-730 (`images_detail`), 734-775 (`image_file`), 791-822 (`apply`), 1007-1020 (`search_options`), 1029-1075 (`threshold_search`), 1089-1137 (`apply_threshold`) (verified)
- Trigger: open the Benchmark page's image drill-down for a run with 100 images: `benchmark.html:883`
  requests `/v1/benchmark/image` once per image, and each request calls
  `BenchmarkResults::load_or_warn` (line 756) which reads and parses `benchmark.json` ("tens of MB",
  live.rs:30) synchronously on a runtime worker thread, then `tokio::fs::read`. Two such requests in
  flight occupy both runtime threads (`runner.rs:52 worker_threads(2)`), so `POST /v1/vision/detection`
  from Blue Iris is not even accepted until they finish. The same happens for a threshold search
  (`search_data` parses results + predictions, then `search::search` runs the exhaustive evaluation),
  for `apply`/`apply_threshold`/`settings` (`save_config` -> file write + `apply_config` -> `reload_model`
  -> `plan_model`, which can take the runtimes mutex), and for `start` (`images::validate` stats
  directories; `benchmark.start` builds the run context).
- Evidence: `pub(super) async fn image_file(...) -> Response { ... let path =
  BenchmarkResults::load_or_warn(&results_path(&state.config_path)) ...` with no `super::blocking`;
  contrast `page`/`status` which wrap their bodies in `super::blocking(move || ...)`.
- Proposed fix: wrap the bodies of these eight handlers in `super::blocking` (they already clone
  `state`), and serve drill-down images from a per-results cached `HashMap<(set,file), PathBuf>`
  (keyed by the same file stamp as `SAVED_CACHE`) instead of re-parsing the results per image.
- Phase: 3

### SV-07  `/v1/fs/browse` walks the filesystem and probes every Windows drive letter on the async thread
- Severity: operability
- Location: src/server/browse.rs:96-105 (`browse` handler), 162-168 (drive probing), 226-258 (`read_dir` + `metadata` per entry) (verified)
- Trigger: on the primary Windows target, click Browse… on the Config page: `roots()` calls
  `PathBuf::from("A:\\").is_dir()` … `"Z:\\"` on every request; a floppy/optical drive letter or a
  disconnected mapped network drive makes `GetFileAttributesW` block for seconds (SMB reconnect
  timeouts tens of seconds). Browsing a directory on a network share or with thousands of entries
  blocks similarly (`metadata` per entry, up to 2000+). All of it runs on one of the two runtime
  threads, stalling detections meanwhile.
- Evidence: `pub(super) async fn browse(State(state): State<Arc<AppState>>, Query(q): Query<BrowseQuery>)
  -> Response { let cfg = state.config_read().clone(); match list(&cfg, q.path.as_deref(),
  q.kind.as_deref()) { ... } }` — no `blocking`.
- Proposed fix: `blocking(move || list(&cfg, ...)).await`; cache `roots()` for a few seconds, and on
  Windows enumerate drives with `GetLogicalDrives()` and skip `DRIVE_REMOVABLE`/`DRIVE_REMOTE` probes
  (or probe them with a short timeout) instead of `is_dir()` per letter.
- Phase: 3

### SV-11  `live.js` polls forever and swallows non-JSON/403 answers; `waitRestart` never gives up
- Severity: operability (low)
- Location: assets/live.js:19-26 (`poll`), 73-79 (`waitRestart`), 404-407 (timers) (verified)
- Trigger: an admin tab left open on a remote machine after `allow_remote_admin` is switched off: every
  2 s `fetch("/v1/config")` succeeds today (open route), but after SV-05 is fixed it would get an HTML
  403 whose `r.json()` rejects and is swallowed by `.catch(function () { return last; })` — the page
  shows stale data with no indication. During a restart that never completes (bind failure of the new
  generation), `waitRestart` retries every 700 ms indefinitely with the "Restarting…" banner pinned.
- Evidence: `busy = fetch("/v1/config", ...).then(function (r) { if (r.status === 304) return last;
  if (!r.ok) throw new Error("HTTP " + r.status); ... }).catch(function () { return last; })`.
- Proposed fix: on `!r.ok`/network error show a small "server not answering (HTTP 403)" note in the
  header and back off (2 s -> 30 s); in `waitRestart` give up after ~2 min with a "still restarting;
  reload manually" message.
- Phase: 4

### SV-12  Admin guard trusts `ConnectInfo` only; reverse proxies on loopback grant admin to everyone
- Severity: operability (documentation)
- Location: src/server.rs:196-205 (verified)
- Trigger: a user fronts the service with nginx/Caddy/Traefik on the same host (common for TLS); every
  proxied request arrives from 127.0.0.1 and is admin. `X-Forwarded-For` is (correctly) ignored, but
  the README's Remote-access section should say that a local reverse proxy is equivalent to
  `allow_remote_admin: true`.
- Evidence: `peer.is_some_and(|a| match a.ip() { IpAddr::V4(ip) => ip.is_loopback(), ... })`.
- Proposed fix: document it; optionally add a `trusted_proxies`/`admin_token` setting for proxied
  deployments (an `Authorization: Bearer` check on the admin router would also close SV-01 for scripts).
- Phase: 4

### SV-15  Handler panics close the connection without a response; the `blocking` doc claims a 500
- Severity: operability
- Location: src/server.rs:358-369 (doc + `resume_unwind`) (verified)
- Trigger: any panic inside a `blocking` closure or handler (e.g. a template render bug, an
  `index out of bounds` in a page builder): tokio catches the task panic, hyper drops the connection,
  Blue Iris (pooled connection) sees a reset on its next request; the panic message is logged by
  tokio's default hook but the process keeps serving. No `CatchPanicLayer` is installed.
- Evidence: doc comment "A panic in `f` is resumed in the handler (axum answers 500 as for any handler
  panic)"; `grep -rn CatchPanic src/ Cargo.toml` is empty.
- Proposed fix: add `tower_http::catch_panic::CatchPanicLayer` (JSON 500 with `success:false`) or
  convert the `JoinError` into a 500 `Response` in `blocking` instead of `resume_unwind`; fix the doc.
- Phase: 1

### BE-11  Layout assumptions on `openvino` crate types are enforced by size only, with a non-exact version pin
- Severity: maintainability
- Location: src/backend/device.rs:618-627 (`raw_request` transmute_copy), src/backend/libs.rs:331-345 (`core_set_property_variadic` transmute_copy), Cargo.toml:37 (`openvino = { version = "0.11" ... }`) vs :40 (`openvino-sys = "=0.11.0"`)
- Trigger: `cargo update` picking a hypothetical `openvino 0.11.1` that adds a field (e.g. a `PhantomData` is fine, a second pointer is not) changes the layout; the `size_of` assert catches a size change but not a reordering/offset change in `repr(Rust)`.
- Evidence: `const _: () = assert!(size_of::<openvino::InferRequest>() == size_of::<*mut ov_infer_request_t>()); ... unsafe { std::mem::transmute_copy(r) }` (device.rs:620-626).
- Proposed fix: pin `openvino = "=0.11.0"` like `openvino-sys`, and prefer an accessor if the crate exposes one (`Tensor::as_c_ptr` exists in 0.11; check `InferRequest`/`Core` or upstream a getter).
- Phase: 5

### BE-12  Stale phase comment on `OrtProbe`
- Severity: maintainability
- Location: src/backend/select.rs:144-145
- Trigger: reading the code.
- Evidence: `/// What ONNX Runtime offers. Phase 6.3 fills this from the loaded library; until then it is /// [`OrtProbe::not_in_build`].` — `ort.rs::probe_providers` fills it today.
- Proposed fix: replace with "filled by `ort::probe_providers`; `not_in_build` without the feature".
- Phase: 5

### BM-07  Unchecked integer arithmetic on values deserialized from benchmark.json
- Severity: maintainability
- Location: src/benchmark/cli.rs:1654, src/benchmark/mod.rs:151 and 430
- Trigger: a hand-edited `benchmark.json`. An agreement with `matched`/`only_device` near `usize::MAX` overflows `agreement_text` (CLI table and the export.rs device row). A stored pred bbox like `1e30` becomes `usize::MAX` via `as usize`, and the area product in `iou` overflows when `regrade` triggers `finish` then `run_agreement`. Debug builds panic; release builds wrap silently, giving wrong text or IoU.
- Evidence:
  ```rust
  let total = a.matched + a.only_device.max(a.only_reference);
  let area = |d: &Detection| (d.x_max.saturating_sub(d.x_min) * d.y_max.saturating_sub(d.y_min)) as f32;
  ```
- Proposed fix: use `saturating_add` in `agreement_text`. Compute the IoU area in f64 from the differences (or saturate the product).
- Phase: 5 (maintainability)

### BM-09  `pub fn percentile` panics on an empty slice
- Severity: maintainability
- Location: src/benchmark/mod.rs:109-113
- Trigger: any future caller passing `&[]`: `rank.clamp(1, 0)` panics (`min > max`) before the index. The only caller today (`Stats::from_samples`) guards it.
- Evidence:
  ```rust
  let rank = ((p / 100.0) * n as f64).ceil() as usize;
  sorted[rank.clamp(1, n) - 1]
  ```
- Proposed fix: return `Option<f64>` (or 0.0 for an empty slice, like `Stats::default`), or make it private.
- Phase: 5 (maintainability)

### BQ-11  NaN scores hang `auc()` and break the binary searches of the sweeps
- Severity: maintainability (latent)
- Location: src/benchmark/roc.rs:89-103; src/benchmark/metrics.rs:895; src/benchmark/roc.rs:180
- Trigger: a NaN in `auc` input: the inner `while` never advances (`NaN == NaN` is false), so `i = j` repeats forever. A NaN confidence in `Breakpoints`/`FrameSteps` sorts first under `total_cmp` and makes `partition_point(|s| s.0 >= t)` non-monotone (counts 0 at every threshold). Not reachable today: post-processing drops NaN (`meets()` uses `>=`) and `frame_samples` folds with `f32::max`; JSON cannot carry NaN.
- Evidence: `while j < s.len() && s[j].0 == s[i].0 { ... j += 1; } ... i = j;`
- Proposed fix: start the group at `j = i + 1` (or compare with `total_cmp(..).is_eq()`), and filter non-finite confidences in `Breakpoints::new`/`FrameSteps::new`/`auc`.
- Phase: 5 (maintainability)

### BQ-12  Non-finite grade fields round-trip as `null`
- Severity: maintainability (latent)
- Location: src/benchmark/grade.rs:226-229
- Trigger: a NaN/inf `p50_ms` or `accuracy_score` (serde_json writes non-finite floats as `null`): `p50_ms: null` fails to deserialize (`f64` from null), making benchmark.json unloadable; `accuracy_score: null` reloads as `None`, turning a graded entry into an unscored one. Not observed: p50 comes from `percentile` over finite timings.
- Evidence: `pub accuracy_score: Option<f64>, ... pub p50_ms: f64,`
- Proposed fix: clamp/validate in `Grades::with_metric` (non-finite p50 -> F with a finite sentinel; non-finite score -> None), or deserialize `p50_ms` with a null-tolerant default.
- Phase: 5 (maintainability)

### BQ-13  Search request: duplicate model names and CSV escaping
- Severity: maintainability
- Location: src/benchmark/search.rs:504, 827-831, 887-893
- Trigger: form `model=a,b,a` or `model=YOLO&model=yolo` searches the model twice (`dedup` removes only consecutive duplicates, and names are not normalized); in the CSV, `device` and `objective` are written unescaped and model names starting with `=`/`+`/`-`/`@` are not neutralized for spreadsheet import.
- Evidence: `names.dedup();` / `r.device,` / `fn csv_field(s: &str) -> String { if s.contains([',', '"', '\n']) ...`
- Proposed fix: dedup by `registry::normalize_name` with a seen-set; pass every text field through `csv_field` and prefix a `'` for leading formula characters.
- Phase: 5 (maintainability)

### CO-15  No crate-level lint policy, so "no unsafe outside backend/" is not enforced
- Severity: maintainability
- Location: src/lib.rs:1-50 (verified, no `#![...]` attributes); Cargo.toml has no `[lints]` table (grep). src/bin/blue_onyx_prism_service.rs:67 (macro-generated unsafe).
- Trigger: any new `unsafe` block or `unwrap` outside `backend/` compiles without a warning. CI only runs `clippy -D warnings` with default lints.
- Evidence: `grep -n '^\[lints' Cargo.toml` returns nothing, and `grep '#!\[' src/lib.rs src/bin/*.rs` returns nothing.
- Proposed fix: add `[lints.rust] unsafe_code = "deny"` with `#[allow(unsafe_code)]` on `backend` modules and the service macro, and consider `clippy::unwrap_used = "warn"` for non-test code.
- Phase: 5

### CO-16  The confidence-threshold rule differs by path: 0 is allowed globally and in `validate_models`, but rejected per model by the benchmark apply
- Severity: maintainability
- Location: src/config_form.rs:120 (`0.0..=1.0`), src/config.rs:613 (`0.0..=1.0`), src/config_form.rs:314 (`> 0.0`) (verified)
- Trigger: a `confidence_threshold` of 0 saved through the form or the models JSON is accepted (every box is reported), while `apply_model_thresholds(…, 0.0)` rejects it.
- Evidence:
  ```
  if !(t.is_finite() && *t > 0.0 && *t <= 1.0) {
  ```
- Proposed fix: define a single `fn check_threshold(field, t)` with one documented range and use it everywhere.
- Phase: 5

### CO-17  Stale docs: the merge rule in config.rs and the README config table
- Severity: maintainability
- Location: src/config.rs:2-3 (verified); README.md:202-228
- Trigger: config.rs says "a CLI value overrides the file value only when it differs from the built-in default", but `merge_cli` overrides whenever the flag is given. The README table omits `preprocess_threads`, `inflight`, `ort_sessions`, `hardware_profile`, `benchmark` and per-model `inflight`, all of which config.rs documents as user-settable.
- Evidence: `//! Merge rule (copied from blue-onyx): a CLI value overrides the file value only when it differs from the built-in default`
- Proposed fix: update the module doc to "a given flag overrides", and add the missing rows to the README table.
- Phase: 5

### NR-22  Unused dependencies in Cargo.toml
- Severity: maintainability
- Location: Cargo.toml:58 (`mime`), :64 (`imageproc`), :95 (`approx`, dev), :48 (clap feature `env`)
- Trigger: every build compiles `imageproc` and its subtree (69 lines in `cargo tree -e normal -p imageproc`) for nothing. Supply-chain and advisory surface grows (`cargo deny` covers them).
- Evidence: `grep -rn "imageproc::\|mime::\|approx::"` over src/ and tests/ finds nothing. The only `imageproc` mention is a comment (src/image.rs:4). `cargo tree -i imageproc` shows it only as a direct dependency of this crate. No `#[arg(env = ...)]` appears in src/.
- Proposed fix: Remove `imageproc`, `mime` and `approx`, drop the clap `env` feature, and run `cargo machete` in CI.
- Phase: 5

### NR-23  Stale guidance in CLAUDE.md, an agent definition and a test comment
- Severity: maintainability
- Location: CLAUDE.md:116-117, .claude/agents/porter.md:14-15, tests/backend_cpu.rs:110
- Trigger: agents and contributors follow outdated rules.
  - CLAUDE.md says the "HTTP integration test skips when `models/` is absent". It is `#[ignore]` and hard-fails without fixtures (tests/common/mod.rs:3-6).
  - porter.md: "never copy ONNX Runtime specific code". CLAUDE.md now says to copy blue-onyx's ONNX Runtime usage freely.
  - backend_cpu.rs:110: "ONNX Runtime is not in this build yet". It is a default feature.
- Evidence: quoted text.
- Proposed fix: Update the three texts.
- Phase: 5

### NR-24  Test fixtures leak temp directories and share an ORT cache across runs
- Severity: maintainability
- Location: tests/common/mod.rs:119-123, tests/backend_ort.rs:58, :232-245
- Trigger:
  - Every ignored-test run leaves `bo_test_*` directories (config, compile caches) in the temp dir.
  - On a dev machine, a `REWRITER_VERSION` bump or a re-downloaded IPcam-general model leaves two `IPcam-general-*.onnx` copies in `temp/bop-ort-test-cache/coreml`. `yolo5_on_coreml` then fails `assert_eq!(rewritten.len(), 1, ...)`.
- Evidence: `std::env::temp_dir().join(format!("bo_test_{tag}_{}", uuid...))` with no removal. `cache_dir: Some(std::env::temp_dir().join("bop-ort-test-cache"))`.
- Proposed fix: Return a guard type that removes the directory on drop (like the `TempDir` guard in the extract tests). Give backend_ort a per-run cache dir, or assert "at least one copy whose name ends in `-v{REWRITER_VERSION}.onnx`".
- Phase: 5

### NR-25  Advisory ignores are undated; the toolchain pin lacks components
- Severity: maintainability
- Location: deny.toml:6-11, rust-toolchain.toml:1-2
- Trigger: the ignores for RUSTSEC-2026-0192 (ttf-parser) and RUSTSEC-2024-0449 (unsafe-libyaml) never come up for review. A fresh checkout runs `cargo clippy` or `cargo fmt` on 1.99.0 without the components installed (as noted in the user's agent memory).
- Evidence: the `ignore = [ "RUSTSEC-2026-0192", ... ]` entries carry no date or review-by. `[toolchain] channel = "1.99.0"` has no `components`.
- Proposed fix: Use the `{ id = "...", reason = "..." }` form with a review date in the reason. Add `components = ["rustfmt", "clippy"]` to rust-toolchain.toml.
- Phase: 5

### NR-26  build_bench_sets.py validation uses `assert`; Pillow is unpinned despite the determinism claim
- Severity: maintainability
- Location: scripts/build_bench_sets.py (validate, lines 685-703), scripts/requirements-bench.txt:4, docs/BENCH_IMAGES.md:132
- Trigger:
  - `python -O scripts/build_bench_sets.py --validate` skips every schema, label and bbox check and reports success.
  - A different Pillow version can change `ImageStat` luma at the 50/255 boundary and flip `day`/`night` tags. That contradicts "same seed and pinned sources give byte-identical manifests".
- Evidence: `assert k in m, f"{path.name}: missing {k}"`, `assert 0 <= x0 < x1 <= im["width"] ...`; `Pillow>=10`.
- Proposed fix: Replace the asserts with explicit checks that append to `errors`, and pin `Pillow==<version>`. Alternatively, weaken the doc claim to "given the same Pillow version".
- Phase: 5

### PL-10  `device` RwLock bypasses the poison-recovering helpers
- Severity: maintainability
- Location: src/worker.rs:620-625, src/worker.rs:1123-1125
- Trigger: a panic while the lock is held (none exists today); afterwards `executionProvider` silently reports "" and the loaded device is never recorded, contradicting `sync.rs`'s stated policy.
- Evidence: `device.read().ok().and_then(...)` and `if let Ok(mut d) = self.device.write() { *d = Some(dev.clone()); }`
- Proposed fix: use `read_recover` / `write_recover` from `crate::sync`.
- Phase: 5

### PL-12  The pipeline run loop is only exercised by an ignored integration test
- Severity: maintainability
- Location: src/worker.rs:633-800 (`run`), tests/pipeline.rs:113
- Trigger: any change to `has_room`, `unstick`, the `Select` arms or `answer_remaining` is unverified by `cargo test` (the unit tests cover `Pipeline` bookkeeping only).
- Evidence: `#[ignore = "needs models/ and the runtime libraries; see tests/common"]`
- Proposed fix: add a fake `Backend` (or feature-gated test backend) so a worker can be driven in-process without OpenVINO: FIFO order, prep failure in the middle, timeout before submit, shutdown with busy slots.
- Phase: 5

### RS-13  `write_entry` temp name and tar path errors lose information
- Severity: maintainability
- Location: src/resources/extract.rs:284 (`out.with_extension("partial")`), 373 (`entry.path()?`)
- Trigger: `libonnxruntime.so.1.24.4` is written through `libonnxruntime.so.1.24.partial`, `plugins.xml` through `plugins.partial`; harmless today (sequential, same directory) but surprising for debugging leftovers; a tar entry with an undecodable name fails the whole extraction with the bare `tar` error and no archive name.
- Evidence:
  ```rust
  let tmp = out.with_extension("partial");
  ...
  let path = entry.path()?.to_string_lossy().into_owned();
  ```
- Proposed fix: `let tmp = out.with_file_name(format!("{name}.partial"))`; add `.with_context(|| format!("reading an entry name in {}", archive.display()))`.
- Phase: 5

### SV-16  `message.html` interpolates HTML-escaped values into a JavaScript string literal
- Severity: maintainability
- Location: templates/message.html:13 (verified)
- Trigger: today `refresh_url` and `generation` are server constants (config_page.rs:775-815), so this
  is safe; the first time a request-derived string (e.g. a `back` parameter echoed verbatim) is passed
  as `refresh_url`, Askama's HTML escaping (`&#34;`) is the wrong escaping for a JS string and a
  `"` + `</script>` payload would break out.
- Evidence: `var target = "{{ refresh_url }}", gen = {{ generation }}, ...` inside `<script>`.
- Proposed fix: put the values in `data-` attributes on `#restart-status` and read them with
  `dataset` from a static script, or use `|json` (askama `json` feature) for the literal.
- Phase: 5



## Per-file rubric answers

### From the backend (Fable subagent) report

### src/backend/spec.rs
1. Inputs: the `device` string from config / CLI / UI dropdown. Trimmed, lowercased, split on `:`/`.`; index must be ASCII digits and fit `u32`; runtime/target combination validated by `Device::new`. Complete, except that an index up to `u32::MAX` is accepted (BE-07).
2. Failure paths: none in non-test code; `parse_index` uses `s.parse().ok()`; no indexing, no unsafe.
3. Resources: none.
4. Concurrency: none (pure, `Copy` types).
5. Platform: none.
6. Correctness: table in the module doc matches `parse`; `Display` round-trips (tested). `openvino_device()` gives `GPU.N`, matching OpenVINO naming.
7. API/config stability: canonical lowercase strings are what `DeviceInfo::spec`, `/v1/devices` and `benchmark.json` store; legacy `GPU`/`CPU` still parse. Adding a target requires touching `from_name`, `supported_by`, `takes_index`, `display_name`.
8. Tests: canonical + legacy forms, garbage rejection, helpers, default index, validation. Missing: nothing material.
9. Readability: good; small functions, module table.

### src/backend/plan.rs
1. Inputs: `DeviceSpec` (already parsed), OpenVINO device names from the runtime, the `Selection`. No raw strings parsed.
2. Failure paths: `out[0]` at 119/133 guarded by `out.len() == 1`; `expect("one failure")` at 217 guarded by `failures.len() == 1`. No unsafe.
3. Resources: none; `try_candidates` runs the caller's attempt sequentially.
4. Concurrency: none.
5. Platform: none.
6. Correctness: matches the module doc and CLAUDE.md (`auto` first candidate not a fallback; CPU last). `openvino:gpu.N` is only checked for "some GPU exists", not `GPU.N` (BE-13).
7. API/config stability: `Candidate.fell_back` feeds the `", fallback"` suffix in `executionProvider`.
8. Tests: GPU present/absent, CPU/NPU/auto/ort, ranking from a selection, order of attempts, error listing, ORT-aware CPU fallbacks. Good.
9. Readability: good.

### src/backend/mod.rs
1. Inputs: `CoreOptions`/`OrtOptions` from config (paths), model path from config. Paths are used as given (operator-trusted).
2. Failure paths: no unwrap/index in non-test code; `Runtimes::new_with` converts every runtime init failure into a stored string. `OvBackend::infer_into` drains stale completions.
3. Resources: `OvBackend` owns slots and the compiled model; `Drop` aborts in-flight requests before freeing (device.rs). `Runtimes` holds one OpenVINO `Core` and one ORT library for the process.
4. Concurrency: `Runtimes` is shared behind a mutex by the registry (per doc); compile serialized. `Backend::submit/collect` are per-worker-thread.
5. Platform: `#[cfg(feature = "onnxruntime")]` branches are symmetric; no OS cfgs here.
6. Correctness: `device_class` for a bare `openvino:gpu` can pick the wrong OV GPU's `DEVICE_TYPE` when `load_model` redirected to an Intel `GPU.N` (BE-09). `info().has_gpu` = any GPU-class option runnable, as documented for `canUseGPU`.
7. API/config stability: `RuntimeInfo` is serialized (`openvino_version`, `available_devices`, `has_gpu`).
8. Tests: none in this file; behavior is covered through plan/select tests and the integration tests elsewhere. `Runtimes::new_with` error paths (both runtimes missing) untested here.
9. Readability: good module doc with a usage sketch; long `new_with` but linear.

### src/backend/detect.rs
1. Inputs: sysfs/procfs file contents (Linux), DXGI descriptors (Windows), CUDA driver strings, `system_info::cpu_name()` (macOS). All parsed with `Option`-returning pure functions; unknown values degrade to fewer/less-described GPUs.
2. Failure paths: `unsafe` in `dxgi::adapters` (613-623) and `cuda_driver::query` (706-776), see the unsafe table; `nvidia.remove(pos)` (548) with `pos` from `position`; `n.max(0)` guards a negative device count; `c_string` uses `from_bytes_until_nul` on fixed buffers. `Box::leak` (710) leaks one library handle per `detect()` call; only `hardware()` (OnceLock) calls it in the tree, so once per process.
3. Resources: the CUDA driver library is intentionally never unloaded.
4. Concurrency: `OnceLock` for the cached result.
5. Platform: three detection paths plus a stub; Linux and macOS exercised by tests on this host, Windows/CUDA paths not (see "Not verified"). LUID byte order: `cuDeviceGetLuid` little-endian bytes vs DXGI `LowPart | HighPart << 32` are consistent.
6. Correctness: `guess_discrete` heuristics documented and tested; `attach_cuda` matching order (key, name, sole leftover) documented and tested.
7. API/config stability: `GpuAdapter`/`HardwareInfo` serialize into `/v1/devices` (vendor as display string, `cuda` skipped when None).
8. Tests: parsers, heuristics, sysfs tree, PCI addresses, CUDA attach; not the FFI paths.
9. Readability: good; the FFI is isolated in two submodules.

### src/backend/device.rs
1. Inputs: model file (read by OpenVINO, not parsed here), port names/shapes from the runtime, `gpu_precision` string from config passed straight to `INFERENCE_PRECISION_HINT` (an invalid value is an error from `compile_on`, with context), `cache_dir` from config (must be UTF-8).
2. Failure paths: `inputs[image_idx]` (321, 464) and `img.shape[2..3]` (485) after explicit checks; `model.inputs[idx]` (960) after `image_index`; `tensor.get_data::<T>()` (876/878/884/888) asserts alignment inside the `openvino` crate (dependency panic, see panic table). FFI: `raw_request` transmute_copy (626), `on_infer_done`, `wait_done`, `set_callback`, `last_error` (see unsafe table). `read_output_into` validates element count vs shape (893-896) after the copy, so a mismatch is an error, not UB.
3. Resources: one input `Tensor` per slot reused; extra-input tensors allocated per inference (RT-DETR only) and kept alive in the slot until collected; output buffers recycled (`OutputBuf::f32_mut` keeps capacity; verified in model/mod.rs:206-215). Completion channel is unbounded but at most `slots` messages are in flight per backend. `abort()` cancels+waits before any request is freed.
4. Concurrency: `CallbackCtx.seq` AtomicU64 `fetch_add(AcqRel)` before `infer_async`, `load(Acquire)` in the callback. `CPU_DEFAULT_PRECISION` is a process-wide OnceLock (fine, Core-wide setting). Potential stale completion if `infer_async` fails synchronously (BE-14, unverified).
5. Platform: Apple arm64 uses the variadic `set_property` shim (libs.rs); the `#[cfg]` pair in `core_set_property` is symmetric.
6. Correctness: `f16_to_f32` tested incl. subnormals/inf/NaN; `make_static` only reshapes dynamic dims; `prefer_intel_gpu` redirects bare `GPU` to the first Intel `GPU.N`; `execution_provider()` string matches the documented shape.
7. API/config stability: `DeviceInfo` fields feed `executionProvider`; `spec` strings are canonical lowercase.
8. Tests: dynamic shapes, EP strings, extra-input lookup, half conversion, image-input detection. Untested: callback/slot lifecycle (needs OpenVINO), `read_output_into` mismatch path.
9. Readability: `load_model` is long (~130 lines) but linear; FFI helpers are small and documented.

### src/backend/libs.rs
1. Inputs: env vars `OPENVINO_INSTALL_DIR`, `INTEL_OPENVINO_DIR`, `PATH`/`LD_LIBRARY_PATH`/`DYLD_LIBRARY_PATH`, `ORT_DYLIB_PATH`; config `openvino_dir`, `onnxruntime_dir`, `download_dir`; directory listings of install dirs; `active.txt`/`flavor.txt` contents. `read_active_flavor`/`write_active_flavor` whitelist the flavor (traversal-safe, tested with `../evil`). File names in `cuda-libs` are loaded if they look like a CUDA library family, including unknown families (`zlibwapi.dll`) — anything a writer of that directory drops in gets its initializer run (the directory is populated from a SHA-256-verified archive; writing there implies writing next to the exe).
2. Failure paths: `unsafe` blocks at 129, 272, 308, 343-360, 715, 731, 852, 859, 876 (see unsafe table). Mutexes use `.lock().ok()`/`lock_recover` (no poisoning panic). `const _` size assert for `openvino::Core`.
3. Resources: preloaded libraries (`PRELOADED`, CUDA libs, `library_version` in ort.rs) are leaked by design. `write_active_flavor` writes `active.txt.tmp` then renames.
4. Concurrency: `prepare_environment` is called from several places (mod.rs:135, service, benchmark CLI/service) and mutates env on Windows via `set_var` — the comment's justification (Win32 env block is locked) holds for `std::env` readers; a C `getenv` reader in another thread would read the CRT copy, so no UB. `preload_dependencies` runs once per process (264).
5. Platform: Windows prepends OpenVINO `bin` and `cuda-libs` to `PATH` (288-312), which child processes (model export, `uv`) inherit (BE-15). Unix preloads by absolute path with `RTLD_GLOBAL`. Apple arm64 variadic shim.
6. Correctness: lookup order 1-5 documented and tested; `pick_library` prefers soname-style names; `cuda_preload_order` tested.
7. API/config stability: `active.txt`, `flavor.txt`, `onnxruntime/<flavor>/`, `onnxruntime/cuda-libs` layouts are shared with the download manager and `setup-onnxruntime`.
8. Tests: lookup order, flavor dirs, library picking, CUDA order, empty preload dir, diagnostics. Untested: Windows `PATH` handling, actual preloads.
9. Readability: two clearly separated sections (OpenVINO, ORT/CUDA); good docs.

### src/backend/select.rs
1. Inputs: OpenVINO device names/`FULL_DEVICE_NAME`/`DEVICE_TYPE` strings (runtime), ORT provider availability, `HardwareInfo`. `OvDevice::spec()` parses names through `spec::parse`, so `HETERO`/`AUTO`/`MULTI`/`BATCH` are ignored.
2. Failure paths: `expect("valid runtime/target combination")` (392) — every call site passes a valid pair; `expect("has Intel")` (511) guarded by `has_vendor`. No unsafe.
3. Resources: none.
4. Concurrency: none (pure).
5. Platform: `hw.is_windows()`/`is_macos()` from strings, so testable cross-platform.
6. Correctness vs CLAUDE.md/README: discrete GPUs by VRAM, NVIDIA `cuda -> tensorrt -> directml(Windows)`, Intel Arc `openvino:gpu.N -> directml`, AMD/other `directml`, integrated Intel via OpenVINO else DirectML, CoreML on Apple silicon, `openvino:cpu` else `ort:cpu`, NPU never auto, unsupported compute capability skips CUDA/TensorRT, measured order first with CPU last — all match (tests at 932-1032 confirm the README's "Pascal card -> ort:directml ahead of an Intel iGPU" line). `ov_of` maps the k-th discrete Intel adapter to OpenVINO's k-th discrete GPU (documented heuristic). Stale comment at 144 (BE-12).
7. API/config stability: `DeviceOption`/`Selection` serialize into `/v1/devices` (`spec` as string, `auto` as strings, `auto_measured`).
8. Tests: compute capability gating, ordinal mapping, probe reasons, ONNX sibling lookup. Missing: a Linux Intel-Arc+iGPU ranking case and an AMD-APU DirectML case (both pure and cheap to add).
9. Readability: `Builder`/`Auto`/`GpuRanking` split is clear; module doc is the spec.

### src/backend/ort.rs
1. Inputs: ORT library path (config/env/exe dir), `ORT_LOG` and `BOP_COREML_REWRITE` env vars, `.onnx` model (parsed by ORT), port names/shapes/dimension symbols from ORT, device index from config (BE-07), `cache_dir` from config.
2. Failure paths: `unsafe` in `library_version` (70-96) and `any_loadable` (273); `inputs[image_idx]` (509, 1035) after `find_image_input`/`position`; `img.shape[2..3]` (529) after rank check; `self.slots[slot]` in `abort` (1357) from `0..len`; `info.inputs[i]` (1178) from `extra_input_index`. `read_output_into` (1226-1251) does not check element count vs shape, but both come from the same ORT value so they agree by construction.
3. Resources: one thread + one `Session` per slot; `bounded(1)` job/result channels; `Drop` aborts (waits), drops senders, joins. A session stuck in `Session::run` blocks `Drop` indefinitely (BE-10). Per inference: CHW copy (`buf.extend_from_slice`), `extra.to_vec()`, `image_input.clone()`, `port.name.clone()`, `Vec<SessionInputValue>` — small, acceptable.
4. Concurrency: `LOADED` mutex (one library per process; different path later is warned and ignored). `session_thread` exits when the job sender is dropped or when a result cannot be delivered.
5. Platform: `dep_names` per OS; Windows preloads `DirectML.dll` next to the library by absolute path (good); CoreML/TensorRT cache subdirs; macOS temp-dir fallback for the CoreML rewrite cache (BE-04).
6. Correctness: `full_name` for CUDA/TensorRT uses adapter position, not CUDA ordinal (BE-03); `dimension_overrides` and `unsupported_on` are tested; `device_actual` yields `cuda:0`, `coreml`, matching `DeviceInfo::actual` docs.
7. API/config stability: `executionProvider` text for ORT = "ONNX Runtime <Target> (<name>[, fallback])"; `ENV_COREML_REWRITE` is a documented knob; the CoreML cache file name embeds `REWRITER_VERSION`.
8. Tests: versions, fixed shapes, overrides, RT-DETR/CoreML gating, dependency reasons, extra-value validation, rewrite switch. Untested: session threads, `collect` after a dead thread, `full_name` (BE-03 would have been caught).
9. Readability: `compile` is long (~150 lines) with a nested metadata-session dance; the rest is well partitioned.

### src/backend/onnx_rewrite.rs
1. Inputs: the entire `.onnx` file bytes (operator-chosen, possibly corrupt/third-party). Wire-level parser: every varint read is bounds-checked (`buf.get`, 54-67), length-delimited ranges use `checked_add` + `<= buf.len()` (88-93), fixed32/64 use `get(pos..pos+n)`; groups and wire types 6/7 are rejected; field number 0 rejected. Strings are validated UTF-8 (`as_str`). No recursion into subgraphs (depth is fixed by the message nesting the code walks). All `as` casts on parsed ints (`as i32`, `as i64`) are truncations with no safety impact. Allocation is bounded by a constant factor of the input (BE-06).
2. Failure paths: `expect("8 bytes")`/`expect("4 bytes")` (85/100) infallible after `get(pos..pos+8/4)`; `unreachable!("filtered above")` (1058) truly unreachable; `expect("type checked above")` (837-838) guaranteed by the preceding `scalar` success for the same dtype; `n.outputs[0]` (1106) after `len() == 1`. Arithmetic on parsed i64 values (889, 917, 969) can overflow (BE-05). `&sha_hex[..min(16)]` on ASCII hex.
3. Resources: whole file in memory plus the rewritten copy; cache written to a temp file then renamed (unique name by pid + counter; stray temp removed on failure).
4. Concurrency: `SEQ` AtomicU64 for temp names; two processes racing on the same cache file both rename atomically (last wins, identical content).
5. Platform: `std::fs::rename` over an existing file replaces on Windows and Unix; the macOS-only consumer uses it.
6. Correctness: HardSigmoid/HardSwish/Split/Flatten emitters tested against decoded views; `reencode` byte-identical (tested); unknown fields preserved; external data left alone; names never collide; dropped dead `Split` size initializers only when no subgraphs. Marker metadata appended.
7. API/config stability: `REWRITER_VERSION` in the cache name; `blue_onyx_prism.rewrite` metadata key in the output model.
8. Tests: extensive (varints, truncation, f16 rounding, every rewrite, caching, garbage). Missing: an overflow/huge-dim case (BE-05) and a size-cap case (BE-06).
9. Readability: excellent module doc; long test module is well organized.

### From the pipeline (Fable subagent) report

### src/model/mod.rs
1. Inputs: port specs and output shapes from the model file (via the backend), `preprocessor_config.json` next to a HF repo (`parse_hf_preprocessor_size`, clamped 32..=4096), class names from YAML, `family` from config. `catalog_hint` and static model dims are not clamped (PL-04).
2. Failure paths: `unreachable!()` at 211/223/235 (guarded by the `matches!` just before, infallible), 371 (`Auto` resolved above, infallible). `to_predictions` `round() as usize` saturates; NaN coordinates become 0 and pass the `x2 <= x1` test (PL-02). `detect_family` indexes `s[len-1]`, `s[len-2]` after `len >= 2` check.
3. Resources: none beyond per-call `Vec`s; `OutputBuf` buffers are reused by the backend.
4. Concurrency: pure.
5. Platform: `hf_preprocessor_size` path join is portable; `dir.file_name() == "onnx"` is case-sensitive (a repo dir named `ONNX` on Windows is skipped; harmless).
6. Correctness: `detect_family` checks `last == 6` before the class-count rules, so a 1-class YOLOv5 export `[1, N, 6]` is detected as YOLO26 (PL-05). Letterbox `scale` recorded in `PreprocessCtx` is the ideal scale, not `new_w / w` after rounding (preprocess.rs:120-124): error <= 0.5 px at 720p; acceptable.
7. API: `ModelFamilyKind` serde names are the config contract (`auto|yolo26|yolo5|yolo8|rtdetr|detr|rfdetr`), tested by `family_from_str_round_trips`.
8. Tests: family detection, HF size parsing, letterbox unmapping, filter/clamp. Untested: non-finite coordinates in `to_predictions`, 1-class YOLOv5 detection.
9. Readability: good module docs; `detect_family` heuristics deserve a decision table comment.

### src/model/classes.rs
1. Inputs: YAML text from disk (`classes` config path or `<stem>.yaml` sidecar); any YAML shape accepted (list, index map, `NAMES|names|Names`). Key count and string length unbounded (file is operator-controlled).
2. Failure paths: `expect("bundled coco_classes.yaml is valid")` at 66 on an `include_str!` constant: provably infallible (covered by test `coco`). Index-map keys `as_u64` then sorted; no indexing.
3. Resources: one file read.
4. Concurrency: pure.
5. Platform: `with_extension("yaml")` sidecar lookup is case-sensitive on Linux (`.YAML` ignored).
6. Correctness: index map with gaps collapses to a dense list, shifting labels (PL-08).
7. API: none.
8. Tests: list/map/file/errors/COCO/sidecar fallback. Gap behaviour is asserted as correct in `map_form` (`10: fire hydrant` -> index 3).
9. Readability: fine.

### src/model/detr.rs
1. Inputs: `logits`/`pred_boxes` outputs (shape + length validated at 148-171 before any index), `num_classes` from the class list.
2. Failure paths: `class_map[i % c]` at 185 with `c == 0` would divide by zero, but then `logits.len() == q*c == 0` and the iterator is empty (infallible). `boxes[4*query..4*query+4]` at 200: `query < q` because `i < q*c`, `boxes.len() == 4q` checked. `c as i64` at 174 fine.
3. Resources: `cand` holds up to `q*c` (300*91 = 27k) entries per request at threshold 0; bounded by the model.
4. Concurrency: pure.
5. Platform: none.
6. Correctness: top-Q over (query, class) and the COCO-91 -> COCO-80 remap match the reference post-processors; NaN logits never pass (tested). Inf logits give score exactly 1.0 (stable sigmoid).
7. API: none.
8. Tests: decode, top-Q cap, COCO-91 mapping, 2-D shapes, every error path, NaN. Complete.
9. Readability: good.

### src/model/nms.rs
1. Inputs: detections already thresholded by the families; `iou_threshold` from config `nms_iou` (unvalidated; any value works, NaN keeps everything).
2. Failure paths: `suppressed[i]`, `dets[j]` all within `n`; `flags.next().unwrap_or(false)` infallible.
3. Resources: O(n^2) time and an n-bool vector; n is unbounded when the threshold is tiny (PL-01).
4-5. n/a.
6. Correctness: class-aware greedy NMS, sorted output; `total_cmp` makes NaN scores sort last instead of panicking.
7. n/a.
8. Tests: IoU, same/different class, chain, threshold edge. No large-n test.
9. Readability: fine.

### src/model/preprocess.rs
1. Inputs: decoded RGB (w, h validated against buffer length), model input size (not bounded, PL-04).
2. Failure paths: `tensor: vec![0.0; 3*w*h]` at 53 aborts on a huge model input; slice ranges at 159-163 are provably inside the plane (`pad_x + new_w <= iw`, `pad_y + new_h <= ih`); `chunks_exact(row_bytes)` on a buffer of exactly `new_w*new_h*3`. `round() as u32` at 120-121 clamped to 1..=iw.
3. Resources: resized buffer and tensor reused; `Resizer` scratch reused.
4. n/a.
5. n/a.
6. Correctness: letterbox pad is integer-floored `(iw-new_w)/2`, matching Ultralytics; LUT normalization verified bit-exact by tests.
7. n/a.
8. Tests: letterbox geometry, pass-through, stretch, normalization, buffer reuse, dirty `run_into`. Good.
9. Readability: good.

### src/model/rtdetr.rs
1. Inputs: `labels/boxes/scores` outputs; shapes and lengths validated at 79-112 before indexing.
2. Failure paths: `labels[i] as usize` after `labels[i] < 0` filter (huge ids become `class_N` labels downstream); `boxes[4*i..4*i+4]` within `4n`. `ctx.input_w as i64` fine.
3. Resources: `scratch` Vec allocated per call when labels are i32 (small).
4-5. n/a.
6. Correctness: feeds the input size as `orig_target_sizes` and un-stretches afterwards: consistent.
7. n/a.
8. Tests: order independence, i32/i64 labels, every error branch.
9. Readability: fine.

### src/model/yolo26.rs
1. Inputs: first f32 output; `[1,N,6]` / `[N,6]` and element count validated (`matrix_dims`).
2. Failure paths: `out.as_f32().unwrap_or_default()` at 38 cannot be None (found by `as_f32().is_some()`); `as_chunks::<6>()` exact because `len == N*6`; `cls.round() as usize` saturating. `dims.0 * dims.1` at 63 may wrap for absurd shapes (release) and then mismatch `len`: still an error, not a panic.
3-5. n/a.
6. Correctness: `num_classes` is unused; a class id beyond the list is reported as `class_N`. Non-finite box coordinates are passed on (PL-02).
7. n/a.
8. Tests: mapping, no-NMS, shape errors, empty outputs.
9. Readability: fine; `matrix_dims`/`meets`/`primary_f32_output` are shared helpers living in yolo26.rs (would fit a `common` module).

### src/model/yolo5.rs
1. Inputs: output `[1,N,5+C]`, `C >= 1`, checked against `num_classes` when known.
2. Failure paths: `row[4]`, `row[5..]` safe because `cols >= 6` and `chunks_exact(cols)`. `obj * best` can be `inf` when the model emits inf (PL-02).
3. Resources: `dets` can reach N (25200 for IPcam) at a tiny threshold (PL-01).
6. Correctness: conf = obj * max class score, early-out on obj: matches YOLOv5 `non_max_suppression`.
8. Tests: decode/threshold/NMS, 2-D shape, errors.
9. Fine.

### src/model/yolo8.rs
1. Inputs: output `[1,4+C,A]`, `C >= 1`, checked against `num_classes`.
2. Failure paths: slices at 60, 69-72 within `channels*anchors` (checked length). NaN scores never win (`v > best`).
3. Resources: two `Vec`s of `anchors` per request (8400 * 12 bytes), plus PL-01.
6. Correctness: matches Ultralytics raw-head decoding.
8. Tests: decode/threshold/NMS, errors.
9. Fine.

### src/image.rs
1. Inputs: raw request bytes (JPEG via zune-jpeg, else the `image` crate), client `image_name` (stem only), predictions.
2. Failure paths: `decode_jpeg` checks `rgb.len() == w*h*3`; `w as u32` fine (JPEG dims <= 65535). `fill_rect`/`blend_pixel` clamp to the image, indices `(y*w+x)*3+3 <= len` given the length check at 150-153. `p.x_min as i64` of a `usize` > i64::MAX wraps negative but is then clamped. `quality.clamp(1,100)`.
3. Resources: zune-jpeg default limits 16384 x 16384 (zune-core 0.5.3 `options/decoder.rs:721`); the `image` fallback has only the 512 MiB allocation limit and `to_rgb8` copies (PL-03). `draw_predictions` clones the whole image; `save_annotated` encodes on the caller's (worker) thread.
4. n/a.
5. Platform: `save_annotated` writes `<stem>.jpg` with a client-chosen stem: reserved device names on Windows (PL-09); over-long stems fail with an error that is only logged.
6. Correctness: NaN confidence prints `NaN%` (follows PL-02).
7. n/a.
8. Tests: JPEG round trip, PNG, out-of-range boxes, path traversal in the name. No oversize-image test.
9. Fine.

### src/prep.rs
1. Inputs: request bytes (`PrepJob.image`), `PrepSpec` from the model geometry.
2. Failure paths: `catch_unwind` around decode + preprocess; a panic drops that geometry's preprocessor and answers an error. `lock_recover` on the pool mutex. No indexing.
3. Resources: one bounded channel `threads * 16`; `submit` blocks the worker when full. `TensorPool` keeps up to 32 buffers per tensor length forever (per generation), entries for old lengths are never dropped (PL-04 note). The decoded `RgbImage` (w*h*3) is allocated per request and not pooled.
4. Concurrency: pool threads exit on channel disconnect; `Drop` joins them (a job in progress completes first). Reply is `try_send` on a bounded(1) channel: cannot block.
5. n/a.
6. Correctness: matches the plain preprocessor bit for bit (tested).
7. n/a.
8. Tests: equality with the preprocessor, bad image, recycling, shutdown. Good.
9. Good.

### src/worker.rs
1. Inputs: `VisionDetectionRequest` (bytes, `image_name`, `min_confidence`), live settings (config), model outputs, `queue_size` (config `worker_queue_size`, unvalidated: PL-07).
2. Failure paths: `unreachable!` at 939 (a `Prep` head is resolved or the loop `break`s first) and 951 (`Ready` always has a slot or `break`s): provably infallible; `slot.unwrap_or_default()` at 953 same. `p.slots[i]` at 1000-1003, 1020: `Done::Slot(i)` indices come from `free_slot` and `resize` is only called before any job exists. `debug_assert_eq!(i, i_tick)` at 772 only in debug. `self.device.write()`/`.read().ok()` at 620-625, 1123 silently skip on poison (PL-10). `ms()` saturates; `as_millis() as u64` fine.
3. Resources: work channel bounded (`queue_size` or 64), soft capacity from the warm-up; `has_room` keeps at most `slots + min(prep_threads,4)` requests in preprocessing; one `bounded(1)` reply channel and one `Meta` per request. On exit: `abort()` on busy slots, every pending request answered. Engine (CompiledModel/ORT sessions) dropped when `run` returns.
4. Concurrency: `runtimes` mutex held for the whole compile (1166-1169; seconds to minutes). Lock order: `runtimes` -> nothing else; `device` RwLock and `LoadGate` never nested under it. `unstick` turns a lost completion into a blocking `collect` (PL-06). `LiveSettings` snapshot per request via `ArcSwap`.
5. Platform: none.
6. Correctness: FIFO reply order; timeout is checked on accept and again before submit, not after inference (the server has its own timeout + grace). `effective_threshold` accepts any positive finite value (PL-01). `error_response` always says `command: "detect"`; the server rewrites it for custom routes.
7. API: response fields set at 1296-1308 match `api.rs`; `analysisRoundTripMs` is 0 (server fills it? not in this file; unverified).
8. Tests: queue sizing, threshold override, gate, lazy trigger idempotence, `Pipeline` slots. The run loop itself is covered only by the ignored `tests/pipeline.rs`.
9. Readability: `run` (~130 lines, nested loops + `Select`) and `fill_slots` are dense but commented; `Stage`/`Done`/`Pipeline` naming is clear.

### src/registry.rs
1. Inputs: config (names normalized, duplicates rejected), `benchmark.json` (loaded by `benchmark::load_or_warn`), provisioning plan.
2. Failure paths: `s.by_name[&key]` at 755 after `contains_key` under a *different* lock acquisition (read at 746, write at 754): the reload mutex serializes reloads, and `apply_live` never changes `by_name`, so the key is still present; `workers[i]` at 523 with `i` from `resolve_default` (counts enabled models; `plans` has exactly those) OK. `Arc::try_unwrap(handle).is_err()` only decides whether to cancel.
3. Resources: one worker thread per enabled model, one `stop-<name>` helper per retire (pruned of finished ones), one `PrepPool` per generation (never resized, logged).
4. Concurrency: `slots` RwLock held briefly; `reload` mutex serializes reloads; `runtimes` is only `try_lock`ed by planning except `first_runnable_wait` (PL-11). `shutdown` joins live workers sequentially: a worker mid-compile delays exit by the remaining compile time.
5. n/a.
6. Correctness: default-model resolution and name normalization well tested. A reload while the old worker compiles leaves two compiled models alive until the old one drains (expected).
7. Config: `enabled`, `name`, `default_model`, `worker_queue_size` semantics stable.
8. Tests: normalization, defaults, duplicates, live apply, retire/shutdown join. Not tested: `reload_model` with a real worker (needs runtimes).
9. Readability: long but well sectioned.

### src/registry/plan.rs
1. Inputs: config paths (`data_path`), class YAML, provisioning state.
2. Failure paths: none (`?` inside the closure becomes `PlanOutcome::Failed`).
3. n/a.
4. `first_runnable_wait` blocks on the runtimes mutex (PL-11).
5. n/a.
6. Correctness: interim-device logic documented and tested with a fake.
7. n/a.
8. Tests: good coverage with `Fake`.
9. Fine.

### src/profile.rs
1. Inputs: `hardware_profile` config string, detected hardware, model names.
2. Failure paths: `cores.max(1)`, all clamps; `(vram_mb + 512) / 1024` fine; `gb.round() as u64` fine.
3-5. `Facts::detect` once per process (`OnceLock`).
6. Correctness: matches the module doc; `explicit_settings_override_the_profile` covers the override rules.
7. `GET /v1/devices` `profile` JSON shape (camelCase keys) is a UI contract.
8. Tests: thorough.
9. Good.

### src/sync.rs
1-9. Pure helpers; poison recovery documented as a policy ("short, panic-free critical sections"). The only poisonable sections in my files that are not short are none; `worker.rs` bypasses these helpers for `device` (PL-10). Tests cover poison recovery.

### src/api.rs
1. Inputs: `VersionInfo::parse` (update check text).
2. Failure paths: none reachable.
6/7. `VisionDetectionResponse` serializes exactly `success, message, error, predictions, count, command, moduleId, executionProvider, canUseGPU, inferenceMs, processMs, analysisRoundTripMs` (serde camelCase leaves the already-camel names unchanged); `Prediction` is snake_case `x_max, x_min, y_max, y_min, confidence, label`. Byte-compatible with the CodeProject.AI shape in CLAUDE.md. `confidence: f32` serializes NaN/inf as `null` (PL-02). `count: i32`.
8. Tests: version ordering only; no serialization snapshot test of the response (worth one, it is the contract).
9. Fine.

### tests/postprocess.rs
Covers every family's happy path and shape/length/type errors with synthetic tensors (17 tests, all pass). Missing: non-finite coordinates, 1-class YOLOv5 auto-detection, huge-N NMS timing.

### tests/pipeline.rs
End-to-end FIFO/overload/shutdown test, `#[ignore]` (needs models + OpenVINO). Not run here. It is the only coverage of the worker run loop, `has_room`, `unstick` and shutdown-with-inflight.

### From the resources (Fable subagent) report

### src/resources/mod.rs
1. Inputs: none (module docs + re-exports).
2. Failure paths: none.
3. Resources: none.
4. Concurrency: none.
5. Platform: none.
6. Correctness: docs match the modules; `export` doc says "YOLO26 export" (now every ExportModel; stale wording only).
7. API/config: re-exports `Flavor, LARGE_DOWNLOAD_BYTES, Platform, Resource, ResourceKind, Event, Job, Manager, ...` used by tests and the server.
8. Tests: n/a.
9. Readability: fine.

### src/resources/actions.rs
1. Inputs: resource ids from HTTP forms (`find` is a case-insensitive exact match against the static catalog; `local:<file>` must be a plain file name with `.onnx`/`.xml`, checked at 390-401); `CustomModel.path/name/classes` from the form or CLI: path resolved against the data root, may be any absolute path (RS-16); name rejects `/ \ ? #` and control chars (489-493); classes YAML parsed by `load_class_names` before acceptance.
2. Failure paths: `files[0]` at 298 and 333 (parts are never empty: catalog test; `model_paths` of an export always returns 2) – infallible; `ok[0]` guarded by `len()==1` (543); `unquote` slice `s[1..len-1]` on ASCII quote bounds (435) – safe; no unwrap outside tests.
3. Resources: `remove` deletes `<data root>/<res.dest>` recursively only after `remove_blocked` + `ours()` manifest check (status.rs 489-491); `remove_toolchain` deletes `tools/*` (export.rs); `forget` clears manager statuses.
4. Concurrency: mutates `Config` under the caller's `save_config` discipline; manager/exporter calls are thread-safe handles. `remove` has a TOCTOU between `remove_blocked` and `remove_dir_all` (a download of the same id queued in between is not blocked; the single download thread would then fail on the missing staging dir and retry) – acceptable.
5. Platform: `unquote` handles Windows "Copy as path"; `same_file` canonicalizes; `contains(['/', '\\'])` on both OSes.
6. Correctness: `download` marks a flavor active only when none is active and ORT is not pinned (174); `add_to_config` paths relative to data root like `download.rs`. OK.
7. API/config: `ActionError::status()` 400/404/409/500 mapping is what the UI expects; `CUSTOM_MODEL_ROW` id used by templates.
8. Tests: 13 unit tests cover custom-model add (dir, quotes, duplicates, classes discovery, names); `download`/`remove`/`export` actions are tested from `src/server/resources.rs` (not in scope) – `remove` of an ORT flavor clearing `active.txt` (242-247) is not covered here.
9. Readability: `add_custom_model` is ~120 lines but linear; good doc comments.

### src/resources/catalog.rs
1. Inputs: none at run time except lookup strings (`find`, `model`, `export_model`, `bench_set`) from CLI/HTTP; all use `eq_ignore_ascii_case`, `strip_prefix_ci` is char-boundary safe (2140-2147, tested), `model()` guards `is_char_boundary` (2183).
2. Failure paths: `panic!("URL is not UTF-8")` in `const fn wheel_file_name` (783) – evaluated at compile time only; `MODEL_INPUT_SIZES`/`F32_MODELS` consistent with CLAUDE.md (RF-DETR L, DEIMv2 L/X); `Resource::size()` sums u64 (max ~1.9 GB).
3. Resources: `bench_sets()` leaks strings once via `OnceLock` (intended `'static`).
4. Concurrency: pure data.
5. Platform: `Platform::current()` from `std::env::consts`; `TOOLS` also lists `macos/x86_64` uv (1967) that no lock/estimate supports – dead data, harmless (`unsupported_reason` refuses).
6. Correctness: every `Part` has url, sha256, size by construction; tests assert 64-char lowercase hex, `https://`, size>0, url ends with file name (tests/resources.rs 73-117); HF URLs pinned to commits; `deimv2!` version `deimv2-add5bcd+hf-<rev>` matches `DEIMV2_COMMIT` prefix. `export_recipe` keyed on name prefix (yolo/rfdetr/deimv2) – the test asserts every EXPORT_MODELS entry has one. Lock files referenced by export.rs exist (`scripts/model-export/locks/{macos-arm64,torch-cpu-index}.txt`, 1262/1271 hashes).
7. API/config: `ResourceKind`/`Layout`/`Flavor` serialize into `/v1/resources` JSON (kebab/lower case); resource ids (`model:<name>`, `onnxruntime-<flavor>`, `tool:uv`) are persisted in `.installed.json` – renaming any breaks "installed" detection.
8. Tests: shape/consistency tests in-module and in tests/resources.rs; URL liveness only via `list-resources --check-urls` (weekly CI, not run here).
9. Readability: long but tabular; macros keep rows uniform.

### src/resources/commands.rs
1. Inputs: `FetchRequest.resources` ids from CLI → `catalog::find` (unknown → error listing ids); config.
2. Failure paths: none (`?`/`bail!`); `remote_size` body count capped at 64 MiB (378-405).
3. Resources: `check_urls` builds one client with 120 s total timeout and a one-byte range GET per URL; `install_all` blocks until done.
4. Concurrency: CLI, single-threaded apart from the manager thread.
5. Platform: none.
6. Correctness: exports are excluded from `--all-for-platform` and run after downloads (CLAUDE.md "user-triggered only"); `fetch_jobs` dedups by key; `apply_ort_activation` re-run after downloads.
7. API/config: text output only.
8. Tests: `fetch_jobs_follow_the_resolver`, `fetch_routes_yolo26_to_the_exporter`; `check_urls`/`remote_size` untested (network).
9. Readability: fine.

### src/resources/export.rs
1. Inputs: catalog resource (static); child-process output lines (ANSI stripped, truncated to 300 chars for the UI log, 200 for `detail`); the exporter's output files (`install_outputs` checks `.onnx` ≥ 1 KiB and exactly 80 `NAMES`); env marker JSON on disk (`serde_json ... .ok()`, a corrupt marker = rebuild). Hashes before execution: `tool:uv` (catalog SHA-256 via the manager), lock file (embedded, `--require-hashes --no-deps --only-binary :all:`), weights and `tool:deimv2-src` (catalog SHA-256), script embedded in the binary and rewritten before each run; `env_ready` compares the marker's `lock_sha256` to the embedded lock. Script cross-check: Rust passes `--recipe {ultralytics|rfdetr|deimv2} --name --family {yolo26|yolo8|rfdetr|rtdetr} --weights --imgsz --no-openvino --out-dir --work [--source]`; `scripts/export_model.py` accepts exactly these (`RECIPES`, `FAMILIES` at 64-65, argparse 120-146), writes `<out>/<name>.onnx` + `<name>.yaml` (369-371, ~491-499, 684-692), stamps `COREML_CACHE_KEY` (280-294), exits non-zero via `SystemExit(str)` on failure which `SystemRunner` reports with the last 8 lines. `--imgsz` for DEIMv2 must equal `config.json` `eval_spatial_size` or the script fails (script 613-624) – consistent for atto/femto (320/416 in `MODEL_INPUT_SIZES`); pico/n/s/m/l/x fall back to 640 (unverified against each config.json; a mismatch fails cleanly at stage Export).
2. Failure paths: `.expect("spawning the export thread")` (1442) – controlled abort at exporter start; `Stage::overall` arithmetic bounded (`p.min(100)`, 464-468); `(b*100).checked_div(t).unwrap_or(0) as u8` (951) truncates only if b>t; `child.try_wait()?` (655); no indexing on untrusted data.
3. Resources: one thread, unbounded crossbeam queue of requests (UI refuses duplicates per id; at most 40 export ids); per child two reader threads + unbounded line channel drained continuously; `tail` bounded to 8, UI log to 40 lines; line length unbounded (RS-08); staging `models/.<name>.export-staging` and `work/<name>` removed on every exit path; `uv-cache` deleted after install; child killed on cancel/timeout but not its descendants (RS-02).
4. Concurrency: `statuses`/`current`/`cancelled` mutexes via `lock_recover`, short sections; `cancel()` reads `status` then `current` (no nested locks); `wait()` on a Condvar with timeout; `ExportLock` is an OS file lock polled every 250 ms honoring cancel; `shutdown()` sets stop + current flag then joins (bounded by the 100/200/250 ms polls).
5. Platform: `CREATE_NO_WINDOW` on Windows; `Scripts/python.exe` vs `bin/python`; `--no-registry` on Windows; `File::try_lock` on all OSes; `uv` is chmod 755 only on Unix (extract.rs).
6. Correctness: stage order and percent monotonic (tested); legacy `yolo26-*` dirs removed; manifest version = resource version so a catalog bump makes "needs export"; `install_outputs` non-atomic across two files (RS-09).
7. API/config: `ExportState`/`ExportStatus` serialize into `/v1/resources` rows (`state`, `percent`, `log`); `ENV_MARKER` JSON shape (`lock`, `lock_sha256`, `python`, `uv`) – changing fields forces a rebuild (benign); manifest in `models/.<name>.installed.json`.
8. Tests: 13 unit tests with a fake runner/fetcher (stages, failures, cancel, callbacks, DEIMv2 source, outdated manifests, legacy cleanup, real `SystemRunner` with `/bin/sh` incl. timeout/cancel, env sanitizing, lock/requirements consistency). Untested: process-tree kill (would fail, RS-02), long-line output, concurrent `ExportLock` between processes, Windows runner.
9. Readability: `run_stages` is ~330 lines; the `enter!` macro and stage comments keep it followable. Module doc is excellent and accurate.

### src/resources/extract.rs
1. Inputs: archive entry names, types, link targets and bodies (untrusted). Names: `components()`/`strip_top()` normalize `\`→`/`, drop empty and `.`, reject `..` and `:` (so `C:` and `file:` prefixes); destination paths are rebuilt from whitelisted pieces, never the entry path; Windows reserved names/trailing dots are not possible because every written name is from a whitelist or a `lib*.so*`/`*.dll`/`*.py`/`*.yml` with plain ASCII components (`source_wanted` 113-118, `wheel_wanted` 791-797). Zip symlinks/dirs skipped on central-directory metadata (329); tar: only regular files written, sym/hard links recreated only when both ends are whitelisted same-directory names (`same_dir_link_target` 255-276); link targets with `/` are rejected for symlinks, hard-link targets go through `strip_top`. Modes (setuid/setgid/exec) are never copied – `write_entry` creates fresh files with default perms; only `uv` gets 0o755 (`uv_check`). Duplicate entries: last wins (same whitelisted name).
2. Failure paths: `entry.path()?` (373) aborts the extraction on a non-UTF-8/invalid tar path (safe fail, context lost – RS-13); zip errors wrapped with context; no indexing/unwrap in non-test code; arithmetic only on `u64` lengths returned by `io::copy`.
3. Resources: no decompressed-size ceiling per entry or in total (RS-03); every entry goes through `<out>.partial` + rename; `create_dir_all` only for whitelisted parents; the staging directory is the manager's responsibility.
4. Concurrency: none (called from the single download thread, under the cross-process lock).
5. Platform: `#[cfg(unix)]` symlinks vs copies elsewhere; `uv.exe` vs `uv`; `.dll` matching for Windows selections is case-sensitive except `wheel_wanted` (lower-cases `.dll`); macOS TBB mirroring (`tbb_mirror_plan`) documented and tested.
6. Correctness: `write_entry`'s temp name `with_extension("partial")` strips a real extension (RS-13, cosmetic); link chain resolution bounded to 4 passes; `soname_links` picks the shortest longer name; `finalize` writes `VERSION`/`flavor.txt` and checks the core library exists.
7. API/config: `ORT_FLAVOR_FILE`/`VERSION` marker files are read by `backend::libs` and `resolve::installed_in` (legacy "installed" detection) – renaming breaks old installs' detection.
8. Tests: 15 tests cover every whitelist, `..`/absolute/`C:` rejection, symlink to `../../etc/passwd` dropped, soname links, uv zip/tgz, source tree, macOS TBB plan, warnings per platform. Untested: zip bomb/size limits (none exist), tar hard links pointing outside, a zip with a symlink entry (skipped by metadata, not exercised), Windows copy fallbacks.
9. Readability: good; clear module doc; sections per layout.

### src/resources/manager.rs
1. Inputs: HTTP response status/headers/body from pinned URLs (`check_url` 762-770 allows `https://` plus loopback `http://`); `Content-Range` start parsed and compared to `have` (1120-1133); body capped at `part.size` (1168); final size and SHA-256 verified (1070-1092) before rename; `.installed.json` parsed leniently (`read_manifest` → None on corrupt = not installed); `local_archives` from the CLI (no hash, documented); `Content-Length` is not compared to the pinned size before streaming (RS-05); redirects follow reqwest defaults (RS-01).
2. Failure paths: `.expect("spawning the download thread")` (470) – controlled abort at start; `bytes * 100 / total` guarded by `total > 0` (282, 312); `t - now` guarded (287-289); Condvar poison recovered; worker early-return on runtime/client build failure leaves jobs hanging (RS-04); `f.flush()?` (1180) drops context (minor).
3. Resources: one worker thread + current-thread tokio runtime + one reqwest client (connect 30 s, read 60 s, no total timeout – intended for GB files); unbounded job channel (dedup by key in `enqueue` and the worker); unbounded subscriber channels (events are few); `.partial` kept on interruption for resume; staging dir `<parent>/.<name>.staging` removed on failure; old install renamed aside and deleted after swap; downloaded archives deleted unless `keep_downloads`; cross-process lock file with PID.
4. Concurrency: `statuses` mutex + Condvar; `on_progress` hook called outside the lock; `emit` holds the subscribers lock while sending (unbounded, non-blocking); `enqueue` check-then-send race (two enqueues of the same key before the worker runs) is resolved by the worker's own dedup (688-690); shutdown sets `stop` then sends `Stop` then joins; `stop` is polled per chunk and in the lock loop but not during hashing/extraction (RS-10).
5. Platform: `File::try_lock` (OS lock, released on process death); renames across the same parent (same filesystem) by construction (`sibling`); Windows "file in use" surfaced with a hint (`in_use_hint`).
6. Correctness: steps 1-6 of the module doc are implemented as written; resume without a validator (RS-15); "larger than pinned" is retryable although it can never succeed (RS-05); `State::describe_for` vs `describe` differ in saturating behavior (both safe).
7. API/config: `Manifest { id, version, sha256[], files[] }` in `.installed.json` / `.<model>.installed.json` (read by resolve/status/export/actions); `State` JSON (`state` tag, `bytes/total`, `retry_at`, `attempts`) feeds `/v1/resources` and `/stats`; `active.txt` via `backend::libs`.
8. Tests: resume with Range after a cut connection, hash mismatch deletes and is not retried, lock contention + `install_all`, https-only, local archive install + swap + leftovers. Untested: redirect handling, oversize body, 200 after Range (truncate path), worker-thread death, `put_in_place` merge path into a legacy flat dir, shutdown during a download.
9. Readability: `run_job` ~150 lines with two clearly separated branches; good.

### src/resources/provision.rs
1. Inputs: config, detected hardware, disk snapshot; manager statuses.
2. Failure paths: none (`unwrap_or(State::Queued)` 338; `unwrap_or_default` on durations).
3. Resources: owns the `Manager` and `Exporter` for the process lifetime (dropped with `Process` in `runner.rs`, which shuts both down); `Wait::poll` clones all statuses per poll (small).
4. Concurrency: two `Mutex<Option<Arc<_>>>` lazily initialized under the lock (no double start); the registry polls `Wait::poll` from worker threads – read-only on the manager.
5. Platform: none.
6. Correctness: ordering (models, CPU fallbacks, rest) as documented; `restart_for` rules match the module doc; with `auto_download: false` manual downloads started by hand still count (`in_flight`). Waits are not time-bounded: a permanently failing non-hash download retries forever (RS-06) and a dead manager thread leaves `Pending` forever (RS-04).
7. API/config: `WaitStatus` texts become model state strings shown in the UI.
8. Tests: tests/provision.rs (auto_download off → Failed with command; downloadable device state; the end-to-end download→Ready test is `#[ignore]`d), server tests elsewhere.
9. Readability: good.

### src/resources/resolve.rs
1. Inputs: config model paths (checked with `is_file`), `ORT_DYLIB_PATH`/`onnxruntime_dir`, manifests (lenient), `VERSION` files; pure resolver over data.
2. Failure paths: `config.models[i]` with `effective_default_index()` from `Config` (valid index by construction); no unwraps.
3. Resources: `detect_installed` probes hardware and (with an NVIDIA GPU) loads CUDA libraries by name (`system_cuda_present`) – called per `rows()` request (see RS-14).
4. Concurrency: pure except `detect_installed`.
5. Platform: `/resolve/` HF paths; `.xml` siblings; Windows paths via `data_path`.
6. Correctness: matches the module doc and CLAUDE.md `auto` rules in tests (CUDA compute-capability gating, one flavor per process, pinned ORT, large gating, needs-export never downloaded); `missing_model_message_for` distinguishes exportable vs unknown files.
7. API/config: `Resolution`/`Need`/`ModelPlan` serialize (ids, device strings) for `/v1/resources`; `FETCH_FOR_CONFIG` text appears in model Failed states.
8. Tests: extensive in tests/resources.rs (29 tests incl. platform matrices).
9. Readability: `Resolver::model` is long (~130 lines) but commented; fine.

### src/resources/status.rs
1. Inputs: resource ids from the UI (`find`, exact case-insensitive match), `models_dir` listing (`local_models`: file names surfaced as `local:<name>` ids and absolute `path` strings in JSON), manifests (lenient).
2. Failure paths: `(bytes * 100 / total) as u8` guarded by `total > 0` (539); no unwraps.
3. Resources: `rows()` does `detect_installed` + `resolve::needed` + per export row `toolchain_state` (stat of 14 paths) + `Platform::current` for ~40 export rows, and `local_models` reads the models dir (RS-14).
4. Concurrency: read-only snapshots of manager/exporter state; may race with a concurrent remove (cosmetic).
5. Platform: paths via `display()`; `model_ext` case-insensitive.
6. Correctness: `remove_blocked` covers downloading, loaded runtime, enabled-model use, running export, user-managed dirs; `status_of` prefers active jobs; the `rows` `Failed` arm is skipped when the files are on disk (shows installed).
7. API/config: `ResourceRow` is the `/v1/resources` schema (many fields; `state` string enum `installed|downloading|queued|verifying|extracting|failed|needed|optional|available|exporting|needs-export|unavailable`); `GROUPS` keys drive the template.
8. Tests: one unit test (recommended flags); row states are tested from `src/server/resources.rs` (out of scope).
9. Readability: `rows()`/`export_row()` are long match-heavy functions; acceptable.

### src/download.rs
1. Inputs: model names from CLI → catalog lookup; destination dir.
2. Failure paths: none outside tests.
3. Resources: delegates to `install_all`.
4. Concurrency: n/a.
5. Platform: `canonicalize` for relative config paths.
6. Correctness: `add_to_config` enable rules tested; `print_list` "partial" detection.
7. API/config: writes `ModelConfig { name, path, family, classes, enabled }` into the config.
8. Tests: 5 unit tests.
9. Readability: fine.

### src/update.rs
1. Inputs: GitHub release JSON (`tag_name: String`, `body: Option<String>`); `tag_name` → `VersionInfo::parse` (u8 segments, errors not panics); `body` unbounded (RS-12). Nothing is downloaded or applied; only a comparison is returned.
2. Failure paths: none; `?` inside an async block converted to `success: false`.
3. Resources: a new reqwest client per uncached check with a 10 s total timeout; `resp.json` buffers the whole body (bounded only by the timeout).
4. Concurrency: `tokio::sync::Mutex` held across the fetch so concurrent cold callers coalesce (documented); worst case every `/status/updates` caller waits 10 s.
5. Platform: none.
6. Correctness: TTLs 1 h / 5 min tested; cache keyed on `current`.
7. API/config: `StatusUpdateResponse` camelCase (blue-onyx compatible).
8. Tests: freshness only.
9. Readability: fine.

### src/setup_openvino.rs
1. Inputs: CLI `--archive` path (existence + extension checked), `--version` (only the pin downloadable), `--dest`.
2. Failure paths: `resource.parts[0].url` (116) – catalog parts are non-empty (test-enforced).
3. Resources: delegates to the manager; `dir_size` recursion over the install dir.
4. Concurrency: n/a.
5. Platform: `std::env::consts`.
6. Correctness: local archive installs record the given version without a hash (documented).
7. API/config: none.
8. Tests: URL per platform; version refusal.
9. Readability: fine.

### src/setup_onnxruntime.rs
1. Inputs: `--flavor`, `--dir`; hardware detection.
2. Failure paths: none.
3. Resources: manager.
4. Concurrency: n/a.
5. Platform: `auto_flavor` matrix tested incl. CUDA compute capability gating.
6. Correctness: `--dir` installs flat and does not activate (matches docs).
7. API/config: `flavor.txt`, `VERSION`, `active.txt`.
8. Tests: 4 unit tests.
9. Readability: fine.

### tests/resources.rs
1-9 (test file): catalog shape (hashes hex, https, sizes, pinned commits, large = CUDA libs only), resolver matrices (Windows Intel/NVIDIA/Pascal, macOS, Linux x64/arm64, flavor switching, pinned ORT, explicit specs, disabled models, no-ORT builds), `detect_installed` layouts/manifests, `fetch_jobs`, export pinning and routing, RF-DETR L migration. Uses temp dirs named by PID (not UUID) for two tests – parallel runs of the same binary are serialized by cargo per test binary, fine. No network. Gaps: nothing exercises `manager` against a hostile server beyond the unit tests (redirects, oversize bodies).

### tests/provision.rs
1-9 (test file): local HTTP fixture (threaded per connection); `auto_download: false` → Failed with command; downloadable device-state annotation; the Initializing→Ready download test is `#[ignore]`d and needs real model + OpenVINO files. Gap: the restart-for-runtime path (`restart_for`) is only tested via server tests elsewhere.

### From the server+UI (Fable subagent) report

### src/server.rs
1. Inputs: peer address (from `ConnectInfo`, trusted), `Accept` header (`wants_html`), config
   `allow_remote_admin`/`bind_address`/`port`. Validated enough; a request without `ConnectInfo` is refused.
2. Failure paths: `blocking()` line 364 `resume_unwind` re-raises a handler panic on the connection task
   (there is no `CatchPanicLayer`: the doc comment "axum answers 500" is wrong, hyper drops the
   connection, SV-15); line 366 `panic!` only when the blocking task is cancelled (runtime shutdown).
   `Shared` uses `read_recover`/`write_recover` (poison tolerant). `bind` line 313 `unwrap_or(port)` fine.
3. Resources: two listeners bound once per process; no header-read / idle / request timeouts and no
   connection or in-flight cap on `axum::serve` (SV-04). `DefaultBodyLimit` 32 MiB on every route.
4. Concurrency: `Shared` is a `std::sync::RwLock<Arc<AppState>>` read per request (short); `admin_only`
   reads the config store `RwLock` per admin request. OK.
5. Platform: IPv6 loopback mapping handled (`::ffff:127.0.0.1`); dual listener when bound to `0.0.0.0`
   or `127.0.0.1`. Nothing Windows-specific here.
6. Correctness: route table matches CLAUDE.md. Loopback check is the only admin gate: no CSRF / Origin /
   `Sec-Fetch-Site` / Host validation (SV-02). `/v1/config` is in the open router (SV-05).
7. API: response shapes for 403 are `{success,message,error}` JSON or an HTML page; Blue Iris routes
   untouched.
8. Tests: loopback matrix, 403 shapes, swap of generations, pooled-connection reuse after early answers.
   Untested: CSRF/Origin behaviour (none exists), request timeouts, panic behaviour.
9. Readability: good module docs; router is explicit and easy to audit.

### src/server/apply.rs
1. Inputs: config store snapshot (already validated by `validate_config` on every writer). None untrusted.
2. Failure paths: `t.get_mut(0..1)` line 33 returns `None` on a multibyte first char (no panic);
   `lock_recover` on `apply_lock`. `reload_model` errors are collected, not propagated.
3. Resources: per-model reload spawns a worker thread and a `stop-<name>` helper thread (registry);
   bounded by model count.
4. Concurrency: `apply_lock` serialises `apply_config`/`reload_models`; `sync_config` is called from
   several read handlers (incl. the open `/v1/config`) so an on-disk edit is applied by whichever request
   arrives first, on that request's thread (inside `blocking` for `/v1/config`).
5. Platform: none.
6. Correctness: `applied_revision` Release/Acquire pairing is right. `describe()` wording fine.
7. API: none.
8. Tests: last-write-wins, ETag/304, disk change pickup, log-level revision bump. Good.
9. Readability: fine.

### src/server/benchmark.rs
1. Inputs: urlencoded forms (`model`, `device`, `dataset` incl. `dir:<any server path>`, `repeat`,
   `warmup`, `max_images`, `accuracy_weight`, `threshold`, `all`, `restart`, `add_folder`), query
   `model`/`device`/`set`/`file`, and `benchmark.json` / `benchmark-preds.json` (self-written, but
   tens of MB). `set`/`file` are confined by `ImageSetInfo::image_path` (`safe_file_name` + listed-file
   check, verified in `report.rs:699-704`). `dir:` datasets are arbitrary server directories by design
   (admin only). Numeric fields parsed by `RunRequest::from_form` (tested: `repeat=0` -> 400).
2. Failure paths: `S[0..3]`, `A[0..3]`, `U[0..3]` at lines 344-351 index constant 4-arrays (infallible).
   `w.device.read().ok()?` line 368 skips a poisoned lock. `serde_json::to_value(..).unwrap_or_default()`.
   No `unwrap` on inputs.
3. Resources: `SAVED_CACHE` holds one `Arc<SavedView>` (bounded). `image_file` reads the whole image
   file into memory (bounded by dataset images). `tokio::fs::read` is used for the image, but...
4. Concurrency: `start` (lines 570-614), `settings` (618-660), `apply` (791-822), `apply_threshold`
   (1089-1137), `images_detail` (675-730), `image_file` (734-775), `search_options` (1007-1020) and
   `threshold_search` (1029-1075) run synchronous file I/O, multi-MB JSON parsing, the config save
   (`save_config` -> file write + `apply_config` -> `reload_model`) and the CPU-bound threshold search
   directly on the 2-thread async runtime (SV-06). `page` and `status` correctly use `blocking`.
5. Platform: `results_path`/`preds_path` are next to the config; `dir:` paths displayed with
   `display()`. Nothing platform-specific.
6. Correctness: `apply_threshold` says `restart=1` is ignored; matches tests. `device_apply_outcome`
   only restarts when something stayed pending. OK.
7. API: `GET /v1/benchmark` JSON shape is consumed by `benchmark.html` only; `changes[]` shape by the
   page; `benchmark.json` schema owned by `benchmark/report.rs`.
8. Tests: start/busy/cancel, apply one/all, apply-threshold one/all/pairs, search endpoint incl.
   objectives and bad input, settings persistence. Untested: `image_file` traversal refusal at the HTTP
   layer (covered in `report.rs` unit test), behaviour under a corrupt results file (`load` error -> 500).
9. Readability: long file (1967) but sectioned; handler doc comments are accurate.

### src/server/browse.rs
1. Inputs: query `path` (any string, trimmed, quotes stripped) and `kind`. `path` is resolved with
   `cfg.data_path` (absolute as-is) and `canonicalize`d; there is **no root confinement**: the endpoint
   lists any directory the service user can read, by design (admin, loopback). NUL bytes / overlong
   names fail in `canonicalize` (`InvalidInput`/`ENAMETOOLONG`) -> 500 with the given path echoed
   (self-supplied, no leak beyond the input). Symlinks are followed (canonical path returned).
   Windows drive-relative `C:foo` is not `is_absolute()` -> joined under the data root -> 404.
   Verbatim prefix stripped by `display`.
2. Failure paths: `UNITS[i]` bounded by the loop guard; `rd.flatten()` skips unreadable entries. No
   unwraps. `io_error` maps kinds to 404/403/500.
3. Resources: `MAX_ENTRIES` 2000 cap on entries, but the `read_dir` + `metadata` walk and `roots()`
   (which `canonicalize`s the data root, models dir, `$HOME`, and on Windows probes `A:\`..`Z:\` with
   `is_dir()`) run on the async thread: `browse` (line 96-105) does not use `blocking` (SV-07).
4. Concurrency: config read lock held briefly (cloned).
5. Platform: Windows drive probing (A:/B: floppy, disconnected mapped network drives) can block for
   seconds; `\\?\UNC\` handled in `display`. macOS adds `/Volumes`. Case-insensitive sort by lowercase.
6. Correctness: file-kind filter is by extension; hidden dotfiles skipped; a file path lists its parent
   with `selected`. OK.
7. API: `Listing` JSON consumed by `config.html` only.
8. Tests: start dir, fallback, filtering/sorting, file selection, relative paths, errors, FS root. Good.
   Untested: Windows paths, symlink behaviour.
9. Readability: fine; doc says "discloses as little as possible" which is true for files, not for folders
   (SV-10).

### src/server/config_page.rs
1. Inputs: `/config` form (`FORM_FIELDS`, applied by `config_form::apply_config_form`), `/config/models`
   form (`enabled`, `default_model`, `device.<name>`, `threshold.<name>`, `action`), `/config/loglevel`
   (`level` from body or query), `/config/restart?back=`. Numeric/IP/device fields are validated in
   `config_form.rs`; **`cache_dir`, `log_path`, `save_image_path` are stored verbatim with no validation**
   (`config_form.rs:111-112,137-141`): any absolute path, UNC path, or path with `..` is accepted; the
   detection path then writes JPEGs there (SV-03). `back` is matched against a fixed list. OK.
2. Failure paths: `device_choices` parses `gpu_index` with `unwrap_or(0)`; `serde_json::to_string_pretty`
   `unwrap_or_default`; `chrono` parse falls back to raw text. No panics on input.
3. Resources: `config_template` does `devices_snapshot` (try_lock on runtimes, `detect_installed` fs
   walk, ORT library lookup), `bench_summary` (cached by file stamp), `resources::status::rows`
   (fs). All inside `blocking`. OK.
4. Concurrency: `save_config` holds no lock across `apply_config` except `apply_lock` inside; the
   `schedule_restart` `tokio::spawn` is called from a blocking-pool thread (runtime context is
   available there; works, tested).
5. Platform: `model_path_example` uses OS separators; config path displayed.
6. Correctness: last-write-wins documented and tested. `force_cpu` only applied when the hidden marker
   `inference_settings` is present (so a JSON-only client cannot accidentally clear it). OK.
7. API/config: `ConfigView` field names are the form names and `forms.server.view` keys of `/v1/config`;
   `live.js` relies on them. Renaming breaks the page silently.
8. Tests: save/validation/rejection re-render, models card, log level, restart token, browse buttons,
   resources card, no-JS anchors. Untested: path fields (none exists), CSRF (none exists).
9. Readability: `config_template` is long (120 lines) but linear.

### src/server/detect.rs
1. Inputs: multipart body (`image` bytes up to 32 MiB, `min_confidence` text parsed with
   `unwrap_or(0.0)` = "use configured"), any other text field collected into a `HashMap` (bounded by the
   body limit), path segment `{model}` (percent-decoded, any length up to hyper's ~408 KiB request
   buffer). `image_name` from the part filename is kept verbatim (used by image saving? not in this
   file; see SV-03 note on `save_image_path` owners).
2. Failure paths: line 123 `as_millis().min(i32::MAX) as i32` safe; line 128 `as u64` safe (u128 ms
   fits). `oneshot` receive errors mapped to responses. No unwraps.
3. Resources: the whole image is buffered in memory per request before the queue check (`field.bytes()`
   line 52) — a flood of concurrent uploads is bounded only by sockets × 32 MiB (SV-04). Queue
   admission uses `is_full()` + `try_send` (bounded crossbeam channel). Reply wait = request timeout +
   500 ms grace.
4. Concurrency: fully async; no locks. A lazy model's `trigger_load` is CAS-guarded.
5. Platform: none.
6. Correctness: early answers consume the upload first (tested) except when the `Multipart` extractor
   itself is rejected (wrong content type): the body is not drained, so hyper closes that connection
   (SV-13, minor). Unknown model name is logged at `warn` with the raw, percent-decoded segment (SV-08).
7. API: `VisionDetectionResponse` fields byte-compatible with CodeProject.AI per `api.rs`; `command`
   set to `detect`/`custom`.
8. Tests: unknown model + no default keep the connection; multipart parse via `/test`. Untested: body
   over the limit (expected `Invalid multipart body`), `min_confidence` garbage.
9. Readability: good.

### src/server/devices.rs
1. Inputs: config values only (device string parsed by `spec::parse`, errors shown as text). None
   from clients.
2. Failure paths: `rt.try_lock().ok()?` (poisoned lock -> falls back to startup facts); no panics.
3. Resources: `devices_snapshot` runs `detect_installed` (fs walk under the download root) and an ORT
   library `lookup` (reads dirs, may `dlopen`? no — `lookup` only locates). Inside `blocking` for
   `/v1/devices`, `/`, `/config`, `/benchmark`.
4. Concurrency: `try_lock` on the runtimes mutex so a compiling worker never stalls the page. Good.
5. Platform: hardware detection is in `backend/detect.rs` (not reviewed here).
6. Correctness: `force_cpu_note` text matches the UI tests.
7. API: `/v1/devices` JSON shape (`selection`, `autoPick`, `profile`) consumed by README/curl users.
8. Tests: endpoint shape, profile report, force-CPU note.
9. Readability: fine.

### src/server/live.rs
1. Inputs: `If-None-Match` header (split on commas, compared to the weak ETag); the Models-card form
   (`threshold.<name>` parsed as f32 then range-checked by `validate_config`? — `validate_config`
   checks models/default only; the range check lives in `config_form::apply_model_thresholds` for the
   benchmark path, while `models_card_edit` line 447-451 accepts any finite f32 incl. `1.5` or `-1`
   (SV-14)). `benchmark.json` parsed from disk (self-written).
2. Failure paths: `HeaderValue::from_str(..).unwrap_or(..)` lines 378-380; `w.device.read().ok()`.
   No panics.
3. Resources: `BENCH_CACHE` one entry; the results file is re-read only on stamp change. `payload`
   serialises the entire config, 20 change records and per-model views on every poll (2 s per open
   tab) — fine for a LAN tool.
4. Concurrency: `payload` calls `sync_config` -> may run `apply_config` (model reloads) on the
   blocking pool from the **open** `/v1/config` route.
5. Platform: none.
6. Correctness: ETag is `W/"<rev>-<hash>"`; `x-config-revision` header set. OK.
7. API: `/v1/config` payload includes `"config": cfg` (the whole config incl. paths and
   `allow_remote_admin`), `"path"` (config file path), `changes`, `forms.server.view` — served to
   **any** client (SV-05).
8. Tests: ETag/304, revision/pending/changes, disk changes.
9. Readability: fine.

### src/server/pages.rs
1. Inputs: `/test` multipart (same parser as the API), `/logs.json` query (`after` u64, `level`
   enum, `limit` clamped 1..2000), none for the static assets. The fallback echoes `uri.path()` inside
   JSON (serde-escaped, `application/json`: not an XSS vector).
2. Failure paths: `test_submit` `spawn_blocking(..).await` errors are turned into text; `runtime_text`
   `unwrap_or(v)`. No panics.
3. Resources: `test_submit` decodes, draws and base64-encodes the image (up to 32 MiB) and embeds it in
   the HTML (`data:` URI ~43 MiB worst case). Admin only. `welcome` uses `blocking`; `stats_page`,
   `stats_json`, `test_page`, `logs_page`, `logs_json` are cheap (atomics, `try_lock`, ring clone of
   <= 2000 entries).
4. Concurrency: `ort_version` uses `try_lock`. OK.
5. Platform: none.
6. Correctness: static assets cached 1 h with `?v=` busting; favicon 1 day. `fallback` returns 404 JSON
   for unknown paths; axum answers 405 for a known path with the wrong method (no body).
7. API: `/stats.json` shape consumed by `stats.html`/`welcome.html`; `/logs.json` by `logs.html`.
8. Tests: escaping (`<b>evil</b>` -> entities) on every page, static assets, test page dispatch, logs.
9. Readability: fine. No security response headers anywhere (SV-09).

### src/server/resources.rs
1. Inputs: forms `id`, `confirm_large`, `add_to_config`, `path`, `name`, `family`, `classes`. `id` is
   matched against the catalog / `local:<file>` (traversal refused, tested); `path`/`classes` go to
   `add_custom_model` which resolves absolute paths as given (any readable `.onnx`/`.xml` on the
   machine can be registered; design). `name` is checked for `/ \ ? #` and control chars.
2. Failure paths: `StatusCode::from_u16(..).unwrap_or(500)`; downcasts handled. No panics.
3. Resources: all handlers inside `blocking`. `resources_export` starts the exporter (downloads uv +
   Python + packages and runs them) when `confirm_large` is truthy — reachable by a cross-site form post
   (SV-02).
4. Concurrency: the `OnDone` closure holds a `Weak<AppState>` so a generation can be dropped. Good.
5. Platform: none here.
6. Correctness: `action_response` puts the message in the resource row. OK.
7. API: `/v1/resources` JSON consumed by `config.html`/`benchmark.html`.
8. Tests: download/add/remove, local models, add-by-path, export incl. cancel (fake toolchain).
9. Readability: fine.

### src/server/test_util.rs (test support)
1-9. Test-only: temp dirs under `std::env::temp_dir()` with UUIDs (never cleaned in `test_state`, only
in `resources_state` callers); a tiny blocking TCP file server thread per `resources_state` (never
joined; process exit cleans up). `metrics.models.write().unwrap()` acceptable in tests. Nothing to fix.

### src/logbuf.rs
1. Inputs: every `tracing` event's message/fields (which include client-supplied strings such as the
   unknown-model path segment, form error texts, file paths).
2. Failure paths: `lock_recover` (poison tolerant); `capacity.max(1)`; `entries.drain(..len-limit)`
   guarded by `len > limit`. No panics.
3. Resources: ring bounded to 2000 **entries**, not bytes: each entry keeps the full message `String`
   (SV-08: a 400 KiB model name × 2000 = ~800 MiB).
4. Concurrency: one `Mutex` held for push/since; `since` clones up to 2000 entries under the lock
   (fine).
5. Platform: none.
6. Correctness: `truncated` computed as `after + 1 < first && after < last_seq`; sequence numbers
   survive eviction. Level ordering comment matches tracing's `Level` ordering.
7. API: `LogEntry` JSON (`seq, ts, level, target, message`) consumed by `logs.html`.
8. Tests: eviction, cursor, level filter/limit, layer capture. Good.
9. Readability: good.

### src/metrics.rs
1. Inputs: model names from the **config** (via `registry.rs:278 self.metrics.model(&name)`), never
   from requests: label cardinality is bounded by the set of names ever configured in this process
   (grows with add/remove cycles; acceptable). Labels escaped (`\\`, `"`, `\n`).
2. Failure paths: `buckets[bucket_index(ms)]` index always `< BUCKETS`; `dropped[reason as usize]`
   bounded by the enum. `read_recover`/`write_recover` everywhere (poison tolerant). `min` sentinel
   `u64::MAX` masked by `count()==0`.
3. Resources: `Metrics.models` Vec of `Arc<ModelMetrics>` never shrinks (documented).
4. Concurrency: lock-free histograms (Relaxed is fine for counters); `RwLock<String>` for device texts.
5. Platform: none.
6. Correctness: non-cumulative buckets accumulated at render; `_count` equals `cum`. The `_ms` names
   keep blue-onyx dashboard compatibility (documented).
7. API: Prometheus names/labels are a public contract (`blue_onyx_prism_*`).
8. Tests: rendering incl. escaping, gauges, bucket boundaries, shared entries by normalized name.
9. Readability: good.

### templates (non-Rust rubric: inputs/escaping, inline script/handlers, forms, JS-off behaviour)
- base.html: all `{{ }}` escaped by Askama; one inline `<script src>` (same origin). JS-off: pages
  render fully.
- benchmark.html: `onsubmit="return false"` inline handlers (lines 23, 108) and a large inline script;
  every dynamic value from JSON is inserted with `textContent`/`el()`; ids built with
  `encodeURIComponent`; `img.src` query-encoded. JS-off: the Run form has no submit path (buttons are
  `type="button"`), the Results/Confidence panels show a `<noscript>` note. Acceptable for an admin
  tool; a CSP without `'unsafe-inline'` would break every page (SV-09).
- config.html: Askama-escaped everywhere (export output lines, paths, model names, titles); forms post
  to admin routes without any token (SV-02); `CSS.escape` used for selectors; JS-off: every form posts
  normally to `#fragment` anchors (tested).
- logs.html: entries rendered with `textContent`; capped at `capacity`; JS-off: `<noscript>` note.
- message.html: `{{ refresh_url }}` and `{{ generation }}` interpolated **inside a `<script>` string
  literal** with HTML escaping (SV-16, currently safe because both are server constants).
- stats.html / welcome.html / test.html: all interpolation escaped; DOM built with `textContent`;
  `data:` image from the server's own base64. welcome.html calls `/v1/status/updateavailable` (server-side
  GitHub check, cached 1 h / 5 min on error). No third-party loads anywhere (system fonts, no CDN).

### assets/live.js
Inputs: `/v1/config` JSON (own origin). Failures: every `fetch` has a `.catch` that returns the last
payload or retries (`waitRestart` every 700 ms forever until the generation changes — bounded to one
timer, SV-11 low). Non-JSON responses (e.g. an HTML 403 after `allow_remote_admin` is switched off
while a remote tab is open) reject `r.json()` and are swallowed; the page keeps polling every 2 s
forever (SV-11). Memory: `listeners` grows only at page init; `last` holds one payload. `new Option(..)`
text is DOM text (safe). No third-party network loads. JS-off: nothing runs; pages are server-rendered.

### assets/style.css
No external resources (`url(...)` absent), no `@import`. Clean.

### From the benchmark-core (Opus subagent) report

### src/benchmark/mod.rs
1. Inputs: model files (via `Backend`), image bytes from datasets (decoded by `crate::image::decode`, errors propagate), class YAML (via `resolve_class_names`), config entries, and, on the reload path, `ImagePred` values deserialized from `benchmark.json` (`From<&ImagePred> for Detection`, line 151). Decoder and backend errors become errors. Deserialized bbox floats are not range-checked (BM-07).
2. Failure paths: every index is guarded by construction (see the panic table). There are three weak spots: the `percentile` precondition (BM-09), `all.len() * repeat` and `Vec::with_capacity(repeat)` with an unbounded CLI `--repeat` (BM-04), and `usize` area multiplication in `iou` on a deserialized huge bbox (BM-07). No `unsafe`.
3. Resources: one backend and infer request per device run, dropped at the end of `time`. `Timings` keeps 5 + 4·buckets `Vec<f64>` of `images × repeat` entries. Per image it keeps ≤100 stored preds plus all eval preds (≥0.05, uncapped). `sweep` re-clones every finished `DeviceResult` after each device (line 1830), which is O(devices²) copies of full `per_image`.
4. Concurrency: `Mutex<Runtimes>` is held only for `plan`/`selection`/`compile` (temporaries, lines 805, 847, 1766), never across inference or progress callbacks, so it does not nest with the service's `Inner` lock. Cancel uses an `AtomicBool` with Relaxed ordering, which is enough for a flag. Compile is not cancellable.
5. Platform: nothing cfg-specific. Cache detection compares a flat `read_dir`, and the `"CPU"` string compare in cli `run_job` assumes OpenVINO naming.
6. Correctness: matches the module docs. Pseudo ground truth is built from the reference's *stored* predictions, which are truncated to 100 per image (BM-10). The nearest-rank percentile is correct and tested.
7. API/config stability: `RunResult`, `AccuracyReport`, `Breakdown`, `ImageRun` and `Stats` are persisted. New fields use `#[serde(default)]`, but many fields are required (`device_name`, `input`, `image`, `image_size`, `compile_ms`, `cache`, `warmup_*`, `repeat`, `stages_ms`, `throughput_fps`, `detections`, `AccuracyReport.threshold`...). Removing or renaming any of them makes the whole file unloadable, which leads to BM-01. Non-finite f32/f64 serialize as `null` and do not load back (BM-02).
8. Tests: scoring (real/pseudo GT, scored_labels, frame ROC), Stats, IoU/compare, pseudo-GT device choice, reference pick and sweep failure path are tested. Untested: a real run (needs runtimes), the pseudo-GT cap, huge `repeat`.
9. Readability: good module docs and small functions. `score` (90 lines) and `threshold_sweep` are the longest.

### src/benchmark/cli.rs
1. Inputs: CLI args (clap), the config file, `benchmark.json` and `benchmark-preds.json` beside it. Checked: `--device`, `--threshold-search --iou` (via `SearchRequest::validate`), `--accuracy-weight` (NaN rejected), `--repeat 0`, `--apply*` needing a config file, and `--dataset` ids / `dir:<path>` (passed as `DatasetRef::Id`, resolved and validated in `images::resolve`/`validate`: unknown id gives an error, a missing dir gives an error). Not checked: `--min-confidence` (NaN/inf/negative/>1, BM-02), `--repeat` upper bound (BM-04), `--warmup`, `--threads`, `--max-images` (0 = all, fine).
2. Failure paths: no unwrap/expect outside tests. Integer `+` on deserialized counts in `agreement_text` (BM-07). `BenchmarkResults::load(...)?` on `--threshold-search` turns a corrupt file into a clear error, not a panic.
3. Resources: one `Runtimes` per process. `--report` writes with plain `std::fs::write` (not atomic; the path is user-chosen). `save_results` goes through the atomic tmp + rename in `report.rs`.
4. Concurrency: single-threaded. The CLI and a running server can both read-modify-write `benchmark.json` (lost update; tmp names differ by pid, so no torn file).
5. Platform: none specific. Paths are printed with `display()`.
6. Correctness: `--apply` / `--apply-threshold` reload the config from disk before writing, so the last writer wins. `write_report` re-reads the config after apply, as documented. A corrupt `benchmark.json` is silently replaced on save (BM-01).
7. API/config stability: the JSON `Report` (`--json`) is an ad-hoc shape and is not covered by a schema test. CLI flag conflicts are tested.
8. Tests: argument conflicts, timing defaults, `--apply` without config, threshold search over stored files, apply_thresholds, dataset precedence and agreement texts are tested. Untested: `--min-confidence` validation, `--report` end to end through `run`.
9. Readability: `run` and `RunCtx` are split sensibly. The printing code is long but flat.

### src/benchmark/service.rs
1. Inputs: HTTP form fields (`RunRequest::from_form`). `repeat` is limited to 1..=1000, `warmup` to ≤100, `accuracy_weight` to 0..1, objective and metric are parsed, devices are parsed with `auto` refused, and datasets must be non-empty (ids validated later by `images::resolve`). `max_images` and `models` are unbounded (0 = all images, which is intended). `dir:<path>` from the form can point anywhere the server user can read. This is admin-only (loopback by default) and is listed as a note, not a finding.
2. Failure paths: no unwrap/expect/index in non-test code. The job runs under `catch_unwind`, so a panic becomes `Failed`. All locks use `lock_recover`.
3. Resources: one detached `std::thread` per run (no JoinHandle kept). `Inner.done` keeps the *summaries* (`per_image` stripped) of one run, replaced on the next start, so it is bounded. `finished` keeps full results (per-image and eval preds) for every model until `save`, which grows with images × devices × models (BM-11). The download wait loop has no deadline (BM-06).
4. Concurrency: one-at-a-time guard under `Inner` (tested). `run_id` prevents a stale thread from overwriting state. The cancel flag reset happens after the lock is released (race, BM-05). `Inner` is never held while the runtimes lock is held: progress callbacks fire outside compile. Blocking work is on the benchmark thread. `status()` clones under the lock (summaries only). `wait_idle` sleep-polls; it must be called off the async runtime (caller not in scope).
5. Platform: none.
6. Correctness: a cancelled run still saves finished models. A save failure then masks the cancel as `Failed` (acceptable). `runs_total` counts requested devices even when they are not options, which matches `sweep` (each one counts once via `on_device`).
7. API/config stability: `Status`/`ProgressView` are camelCase JSON for the UI. `RunRequest` is serialized into `Status.request`. `to_config` writes `benchmark.*` keys.
8. Tests: form parsing and bounds, the busy guard, cancel, panic to Failed, and unknown or missing models are tested. Untested: download wait, the cancel race, a save on cancel.
9. Readability: clear. `pseudo_ground_truth` nests three levels of match.

### src/benchmark/current.rs
1. Inputs: the config (validated elsewhere), the loaded `BenchmarkResults` and `StoredPreds` (possibly hand-edited). It only reads them through `points_at` (search.rs, not in scope) and `known_point_at`.
2. Failure paths: none. No indexes or unwraps outside tests.
3. Resources: clones only, used for a view.
4. Concurrency: pure.
5. Platform: none.
6. Correctness: the counts check (`p.at_run.counts == a.configured.counts`) rightly refuses predictions from another run. `strip_suffix` falls back to the unchanged text when the stored recommendation was edited.
7. API/config stability: `now` is never persisted (tested: the saved JSON has no `"now"`).
8. Tests: exact, curve, none and other-run sources, plus from_config, are tested.
9. Readability: good.

### src/benchmark/export.rs
1. Inputs: `BenchmarkResults` (model names, paths, dataset ids and titles, licenses, error strings, runtime and hardware strings, device labels) and the output path from `--report`.
2. Failure paths: `ranking[0]` (guarded at line 425) and `v.resize`. No panics reachable.
3. Resources: builds the whole report in memory and writes it non-atomically to the user-chosen path. The file name comes only from the CLI argument, never from model or dataset names.
4. Concurrency: none.
5. Platform: the extension compare is case-insensitive (tested `.HTML`).
6. Correctness: HTML escapes every interpolated value at the single sink (headings, paragraphs, table titles, head cells, body cells all go through `html_escape`; text-node context only, so `'` does not matter). Markdown escapes only `|` and newline in table cells. Headings and paragraphs are emitted raw, so raw HTML, Markdown syntax and newlines in model names or error strings pass through (BM-08). A backslash before `|` defeats the cell escape.
7. API/config stability: the report is not machine-read.
8. Tests: escaping of a model name in HTML, write by extension, threshold sections.
9. Readability: the table and section model is clean.

### src/benchmark/report.rs
1. Inputs: `benchmark.json` from disk (`load`). JSON errors become errors (tested with garbage). Unknown fields are ignored (forward compatible). Missing required fields or non-finite floats written as `null` make the *whole* file fail. `ImageSetInfo.dir` plus `file` from the file are joined in `image_path`. `safe_file_name` blocks `/`, `\`, `..` and NUL but not `:`, so on Windows a hand-edited `"C:x.jpg"` joins drive-relative. This needs write access to the results file, so it is a note.
2. Failure paths: `per_image[k]` is guarded by the length check (line 110). Division is guarded (`n == 0`, `max(1e-9)`). The `NOISE_MARGIN` multiply is fine.
3. Resources: `save` writes `benchmark.json.<pid>.tmp` and renames it. On a write or rename error the tmp file is left behind. `summary()` clones then strips.
4. Concurrency: there is no file lock. Read-merge-write is not atomic across processes (CLI vs server) or across server handlers that save the same file (server code is out of scope).
5. Platform: `rename` over an existing file works on Windows (MoveFileEx REPLACE_EXISTING); not verified there.
6. Correctness: `recommend` and `speed_order` share `Ranked`, and the tie logic is tested. `merge` drops *every* previous model when `HardwareSummary` differs in any string, including the GPU Display text, which now carries optional CUDA facts. `same_gpus` was made tolerant of that, but `merge` was not (BM-03). A load failure gives `None`, so the next save overwrites (BM-01).
7. API/config stability: `BenchmarkResults` is the UI and `auto` ordering contract. `auto_order` tolerates unparseable specs. An older file without `threshold`/`threshold_objective` loads (tested). Files without `datasets`/`images`/`configured` load via defaults.
8. Tests: round trip, merge, garbage, recommendation and tie rules, regrade, image_path restriction, auto_order GPU matching, and older-results compatibility are tested. Untested: load with a missing required field, non-finite values, cross-process merge.
9. Readability: good. `recommend` is long but commented.

### From the benchmark-metrics (Opus subagent) report

### src/benchmark/grade.rs
1. Inputs: accuracy score / p50 from the run, `Weights` and `AccuracyMetric` from config/CLI, `Grades` deserialized from benchmark.json. Weights are sanitized (negatives clamped, non-finite/zero sum -> default); metric parse is strict.
2. Failure paths: none (no unwrap/index/cast outside tests). `Self::ALL[i]` with `i` from `position` over a 4-array is in bounds. NaN inputs grade F (comparisons false), rank_cmp uses total_cmp.
3. Resources: none.
4. Concurrency: none (pure).
5. Platform: none.
6. Correctness: matches README.md:481-487 exactly (A<50/B<100/C<200/D<400 ms; AP50 blend 0.2 small recall, A>=.70 B>=.60 C>=.50 D>=.40; AUC A>=.95 B>=.90 C>=.80 D>=.70; 60/40 default; ties 0.015/0.01). Doc nuance: README says "blended 20% with small-object recall" without "when the images have small objects" (code: `None => ap50`).
7. API: `Grades`, `Grade`, `AccuracyMetric` are in benchmark.json; `metric` has `serde(default)` for old files (tested). Non-finite f64 fields serialize as `null` (BQ-12).
8. Tests: thresholds at bounds, metric parsing, overall weighting, ranking, old-file default. Untested: NaN/inf inputs, negative weights.
9. Readability: good; module doc is the spec.

### src/benchmark/images.rs
1. Inputs: embedded manifests (trusted, compile time), user dataset dirs from config/UI (`dir:`), `manifest.json`, COCO JSON, YOLO txt + `classes.txt`/`data.yaml` in those dirs, filesystem listings. File names from embedded manifests pass `safe_file_name`; user-dir manifest entries are only matched by name against listed files (no path built from them). COCO/YOLO contents are not range-checked (BQ-08, BQ-09); YAML index map is not bounded (BQ-01).
2. Failure paths: `out[i as usize]` (713) and `vec![...; len]` (711) reachable from a user `data.yaml` (BQ-01). `self.images[i * len / n]` (403) in bounds since i<n<len. COCO `out[i]` (631) index from the map built on the same vec. `npos`-style subtractions: none. `?` errors carry file context.
3. Resources: `SetImage::bytes` reads a whole file with no size ceiling (BQ-05); `image_dimensions` reads headers only; `ImageSet::sample` decodes the embedded image fully just for its size (minor). COCO/YOLO/manifest text read fully (local files).
4. Concurrency: `OnceLock` for builtin manifests; otherwise none.
5. Platform: `safe_file_name` rejects `/` and `\` but not `:` (drive-relative `C:x.jpg`, NTFS ADS) (BQ-03). Extension match is case-insensitive; `list_images` sorts by OS path order; follows symlinks via `is_file`.
6. Correctness: "images not in the COCO JSON have no objects" is silent even when the JSON matches nothing (BQ-08); basename collisions in COCO `file_name` (first wins). YOLO-seg polygon lines are read as boxes (BQ-09). Duplicate doc comment at 233-235.
7. API: manifest JSON shape (`Manifest`, `ManifestImage`, `ManifestObject`) is the contract with assets/bench and catalog resources; `GroundTruth` serde (`kind`/`reference`) is stored in benchmark.json.
8. Tests: buckets, sample, manifest fixture (missing file, no-size image, ignore), limit, COCO parser (crowd, unknown image id), YOLO parser/yaml, dir loading (COCO/YOLO/plain/missing), resolution/validation, embedded manifests vs catalog (URL https, sha len 64, sizes, safe names, COCO-80 labels). Untested: `images/` subdir, `instances*.json` discovery, `data.yaml` in load_dir, malformed YOLO numbers, huge YAML index, Windows drive names.
9. Readability: good; `load_dir` (96 lines) is the longest function.

### src/benchmark/metrics.rs
1. Inputs: boxes and confidences from model post-processing (NaN already dropped upstream: `meets()` is `>=`, src/model/yolo26.rs:44), ground truth from manifests/COCO/YOLO/benchmark.json.
2. Failure paths: `npos - tp` (486, 813, 903, 924) is safe because `SweepImage::with_iou`/`match_image` only ever mark a non-ignored, same-label GT once per TP (provably tp <= npos); the only other `SweepImage` literal is in report.rs tests. `aps[0]` (575) over a 10-element const; `img.gt[*j]` (727) j from matching; `steps[n-1]` with n>0; `pts[(i*last+(max-1)/2)/(max-1)]` (935) max index = last (checked). No unwrap.
3. Resources: `evaluate` clones per class x 10 IoUs x image (allocation-heavy but bounded by dataset size). No unbounded state.
4. Concurrency: none.
5. Platform: none.
6. Correctness: AP 101-point interpolation and envelope match COCO; empty sets give None (npos 0) / Some(0.0) (no detections) as documented; per-class maxDets=100 like pycocotools. Divergence: ignore (crowd) regions use standard IoU instead of pycocotools' intersection-over-detection (BQ-06). NaN in a `Breakpoints` step would break `partition_point` (BQ-11, latent). `decimated(max<2)` returns everything (callers use 50/150/200).
7. API: `Counts` (`fn` rename), `ThresholdPoint` (flattened counts, optional `frame`), `ClassMetrics`, `Summary`, `SizeRecall` are in benchmark.json.
8. Tests: label map, IoU/buckets, greedy matching incl. ignore, AP hand values incl. empty, evaluate, counts, sweep vs re-matching (randomized), breakpoints vs brute force, decimation ends. Untested: crowd-region matching vs pycocotools, degenerate/NaN boxes.
9. Readability: good, long but well documented.

### src/benchmark/roc.rs
1. Inputs: `FrameSample`s built by `frame_samples` (score = fold with `f32::max` from 0.0, which drops NaN).
2. Failure paths: `steps[n-1]` with n>0; `pts[...]` in `thin` bounded as above; `u128` counters cannot overflow; `as f64` fine. `auc` loops forever on a NaN score (BQ-11, latent).
3. Resources: copies sample vectors per class (O(classes x samples)); bounded.
4. Concurrency: none.
5. Platform: none.
6. Correctness: Mann-Whitney with half ties is exact and equals trapezoid (tested); undefined without positives/negatives; macro over defined classes, micro pooled. Matches README.md:472-480.
7. API: `FrameRates`, `RocPoint` (`t`), `ClassRoc`, `FrameRoc` (curve skipped when empty) are stored.
8. Tests: brute-force equality with ties, extremes, sample exclusion, macro/micro, rates, thinning. Untested: NaN scores.
9. Readability: good.

### src/benchmark/search.rs
1. Inputs: urlencoded form from the admin UI (`from_form`: iou validated to (0,1), objective parsed, device parsed); `benchmark-preds.json` and `benchmark.json` from disk (serde-typed; class index checked with `.get`). Model names from the form are not normalized/deduplicated (BQ-13).
2. Failure paths: `self.images[k]` (154) k from enumerate of the same vec; `inputs[k]` indices from `0..inputs.len()`; `as u32` saturating casts (73, 121). No unwrap outside tests.
3. Resources: whole preds file read and parsed per search/options request, no size bound; file grows with models x devices x images x detections (BQ-04). `save` writes `<name>.json.<pid>.tmp` then renames; tmp not removed if rename fails.
4. Concurrency: `search` is CPU work called from async handlers (`threshold_search`, `search_options` in src/server/benchmark.rs:1007/1029 do not use `server::blocking`; cross-file, see BQ-04). Save is single-writer per process (one benchmark at a time).
5. Platform: `rename` over an existing file works on Windows (MoveFileEx replace). Paths built from config dir only.
6. Correctness: matches module doc; any load error of the preds file wipes stored runs of other models on the next save (BQ-02). Stored runs whose images were dropped from `image_sets` are skipped silently (scored_inputs 556-561).
7. API: `benchmark-preds.json` tuple format `[class, conf, x0, y0, x1, y1]`; `StoredRun` fields have no `serde(default)`, so any added field makes old files unreadable, which then triggers BQ-02. CSV column order is an export contract.
8. Tests: hand-computed search, filters, frame ROC/Youden, form parsing, round-trip + merge. Untested: corrupt preds file + save, large files, duplicate model names.
9. Readability: good; `search_run` ~100 lines.

### src/benchmark/threshold.rs (all ranges)
1. Inputs: objective strings from config/CLI/form (strict parse; NaN/out-of-range rejected), `Breakpoints` built in-process, stored `Pick`/`Curve` from benchmark.json.
2. Failure paths: `segments[best_segment(..)?]`, `segs[a]`/`segs[b]`/`segs[best]` indices come from the same slice; `(t*100).round() as u32 + 1` with t<=1. `plateau` indexes guarded by `i + 1 < len`. No unwrap outside tests.
3. Resources: `exact_pick` O(n log n) per objective; up-search at most 100 point evaluations.
4. Concurrency: none.
5. Platform: none.
6. Correctness: plateau/midpoint/round-down logic matches README; when no 0.01 step meets a precision/FPR target the pick still says `met: true` and `apply_value` writes it (BQ-07). `ThresholdAdvice::objective()` silently falls back to F1 on an unparsable stored string.
7. API: `Objective` serializes as a string (`precision:0.9`); `Pick`, `Curve`, `ThresholdSweep`, `ThresholdAdvice` stored with defaults for newer fields.
8. Tests: parsing, frame objectives vs brute force, hand-computed Youden/FPR, grid picks, ties, unreachable targets, table rows, NMS-filter exactness (Yolo5), plateau, exact vs brute force. The precision-miss-with-met case is asserted as intended (1470-1479) — no test of `apply_value` in that case here.
9. Readability: dense but well commented; `exact_pick` ~90 lines with two closures.

### assets/bench/*.json (non-Rust)
1. Shape: all three are full manifest objects (`id,title,description,license,attribution,source,revision,images`; bmd45 and exdark add `scored_labels`) that deserialize into `Manifest`; every image has `file,url,sha256,size,width,height,tags,objects`.
2. Pins: 44/180/160 images, every `url` https (huggingface.co pinned to a commit revision for bmd45/exdark; `s3.amazonaws.com/images.cocodataset.org` for COCO, a mutable URL protected by SHA-256), every `sha256` 64 hex chars, every `size` present (totals 142,388,382 / 31,492,283 / 6,689,717 bytes match the descriptions). Counts match README.md:458-460.
3. Content: no duplicate or unsafe file names; no inverted/negative/out-of-image boxes; labels are COCO-80 spellings (test enforces); `small-objects` tag agrees with "any non-ignored box < 32x32" on every image; `empty` tags are images with no scored, non-ignored object. exdark ignore labels (boat, bottle, chair, cup, diningtable) are outside `scored_labels`, so they are dropped before matching and the ignore flag has no effect (harmless). coco-cctv has 21 crowd ignore regions on 22 `crowd` images, which is where BQ-06 applies.
4. Licensing notes are present (COCO Flickr licenses, CC BY 4.0, ExDark BSD-3 + commercial-use caveat).

### tests/data/bench/** (fixtures)
- manifest-set: manifest with an ignore object, an image with no width/height (exercises `image_dimensions`), and a listed-but-missing file (exercises `missing` count).
- coco-dir: `annotations.json` auto-discovery, one annotated and one unannotated image (empty GT).
- yolo-dir: `labels/` discovery, `classes.txt` names, normalized-to-pixel conversion.
- plain-dir: no ground truth -> `GroundTruth::None`.
Not covered by fixtures: `images/` subdirectory, `instances*.json`, `data.yaml`, malformed label input.

### From the core/cli/config (Opus subagent) report

### src/lib.rs
1. Inputs: none. `exe_dir()` relies on `current_exe`, else cwd, else `""`.
2. Failure paths: none. `unwrap_or_default` can give an empty `PathBuf`, so `resolve_path` would then resolve against the cwd without saying so.
3. Resources: none.
4. Concurrency: none.
5. Platform: `current_exe` on Linux reads `/proc/self/exe`. Through a symlinked binary, the "exe dir" is the target's directory.
6. Correctness: the module map matches the modules. The doc line "runtimes (OpenVINO today)" is stale, because ORT is a backend too.
7. API stability: `MODULE_ID`, `PROCESSED_BY` and `DEFAULT_PORT` are wire constants for Blue Iris and must not change.
8. Tests: none (trivial).
9. Readability: there is no crate lint policy: no `#![deny(unsafe_code)]`, `#![warn(clippy::...)]` or `[lints]` table in Cargo.toml (CO-15).

### src/bin/blue_onyx_prism.rs
1. Inputs: CLI arguments (clap) and the config file. Paths are absolutized against the cwd at line 21. The `--model` paths of `list-devices` are not checked for existence (the plan prints anyway).
2. Failure paths: no unwrap. `std::process::exit(1)` for a failed benchmark (line 334) skips destructors; nothing is held at that point.
3. Resources: one tokio runtime (`runner::build_runtime`). The signal task lives for the process.
4. Concurrency: the signal task cancels `shutdown`, and nothing else is involved.
5. Platform: SIGTERM and Ctrl-C on Unix, Ctrl-C only on Windows (the console binary). A second signal is ignored (CO-12).
6. Correctness: `resolve_config` runs before `init_logging` (lines 346-348), so the legacy-rename info and the "could not save merged config" warning go nowhere (CO-11). `log_path` from the file is passed unresolved (line 348), so it is relative to the cwd, not the exe dir (CO-10).
7. API stability: subcommand names and flags are user-facing. Exit codes are 0 and 1 only.
8. Tests: none for `main` or `run`. Subcommands are covered through `cli` tests.
9. Readability: `run` is a long match (155 lines) but flat. `add-model` and `download-models --add-to-config` duplicate the load-or-default-then-save logic.

### src/bin/blue_onyx_prism_benchmark.rs
1. Inputs: clap args are forwarded as they are.
2. Failure paths: none.
3. Resources: none here.
4. Concurrency: none.
5. Platform: portable.
6. Correctness: thin wrapper. `None` config means the benchmark CLI finds its own default.
7. API stability: binary name.
8. Tests: none (trivial).
9. Readability: fine.

### src/bin/blue_onyx_prism_service.rs
1. Inputs: SCM control codes (130 and 131 user codes) and the service config file.
2. Failure paths: `lock_recover` is used for the token mutex. `checkpoint += 1` (u32) cannot overflow in practice. No unwrap. `define_windows_service!` expands to an `extern "system"` fn with an `unsafe` argument parse outside `backend/` (CO-15).
3. Resources: one server thread and one runtime per `server_loop`. The stop loop waits for `server.is_finished()` with no upper bound (line 251, CO-12).
4. Concurrency: the handler only cancels tokens and sends on an mpsc. `restart_requested` uses SeqCst. A restart that arrives before `*current = attempt` cancels the previous, already-finished token and is lost. The window is small and not verified.
5. Platform: Windows only. It was not compiled or run on this machine.
6. Correctness: the first start creates a default config with `models: []`, and `runner::ensure_models` then fails before the listener is bound. The service retries every 5 s and never serves the UI that `install_service.ps1` tells the user to open (CO-05). It reports `Running` before anything is bound.
7. API stability: service name `BlueOnyxPrismService`, event source `BlueOnyxPrism`, and control codes 130 and 131.
8. Tests: none (not compiled in CI on this host; Windows CI unverified).
9. Readability: good module docs. `run_service` and `server_loop` are reasonably sized.

### src/bin/test_blue_onyx_prism.rs
1. Inputs: `--url` is used verbatim. `--model` is interpolated into the path without percent-encoding (line 160). `--image` is any file.
2. Failure paths: `partial_cmp().unwrap()` (line 122) is infallible because the values are finite elapsed times. `interval * (i / parallel) as u32` (line 170) truncates usize to u32 and can panic on Duration overflow for a huge `--interval-ms`. `outcomes[0]` is safe because `repeat >= 1`.
3. Resources: the whole image is copied per request (`to_vec`). Fine for a test tool.
4. Concurrency: `buffered(parallel)`. Each future's sleep starts when the future is first polled, not at stream start, so the delays compound (CO-14).
5. Platform: portable.
6. Correctness: the exit code is non-zero on any failure (via `bail!`). `--list` with a non-2xx status prints "one or more requests failed". `--save` with no successful response writes an unannotated copy and says "saved annotated image".
7. API stability: dev tool.
8. Tests: none.
9. Readability: fine.

### src/cli.rs
1. Inputs: CLI flags. `--device` is validated by the spec parser and `--bind-address` by IpAddr. `--confidence-threshold`, `--request-timeout-secs`, `--worker-queue-size`, `--intra-threads`, `--port 0` and `--gpu-index` are not range-checked, and all of them are written back to the file (CO-04).
2. Failure paths: `expect` at line 25 is infallible (PossibleValuesParser). `init_logging` returns an error when `log_path` cannot be created, which aborts the CLI server. The service falls back instead.
3. Resources: `tracing_appender::rolling::daily` at line 501 has no `max_log_files`, so the files grow without bound (CO-06). The default lossy `non_blocking` writer silently drops lines when its 128k-line buffer is full.
4. Concurrency: the reload handle is cloned freely. The appender guard is kept in an `Arc`, so the file is flushed when the last clone drops.
5. Platform: path absolutization is portable. Windows drive-relative paths (`C:foo`) are `!is_absolute()` and get joined onto the cwd; that rule was not reviewed further.
6. Correctness: (a) `--model` "already present" compares the absolutized CLI path with the raw config path (line 385). A relative entry (as `download-models --add-to-config` writes it) never matches, so the whole model list is replaced and saved (CO-01). (b) `resolve_config` rewrites the file on every start with a non-atomic write (line 220, CO-02). (c) Boolean flags are sticky. `--allow-remote-admin` passed once stays on in the file permanently (CO-07). (d) `for_service` creates a model-less config (CO-05).
7. API stability: flag names. The config file shape is written back on every start.
8. Tests: merge rules, absolutization, device validation and `--model` replace/enable are covered. A relative `--model` against a relative config entry, read-only write-back and logging init are not tested.
9. Readability: fine. The `absolutized` command remap is long but mechanical.

### src/config.rs
1. Inputs: the config JSON (admin-owned, but also rewritten by the UI). `Config::load` does serde only, with no range or cross-field validation (CO-04). `validate_config` exists but is only called by `config_form`.
2. Failure paths: `self.models[i]` at line 454 uses an index from `position`, so it is infallible. `round_threshold_f64` uses `as f32`/`as f64` casts on finite values only. `clean_f32` parses with a fallback.
3. Resources: none. `save` uses `std::fs::write` (truncate then write, no temp file and rename, no fsync) (CO-02).
4. Concurrency: none (pure data).
5. Platform: `migrate_legacy` falls back to the legacy path when the rename fails (read-only dir), which is good. Its log lines are lost at CLI startup (CO-11). For a missing directory both paths are absent and the current name is returned.
6. Correctness: the module doc (lines 2-3) says a CLI value overrides "only when it differs from the built-in default", but `merge_cli` overrides whenever the flag is given (CO-17). `validate_models` allows a per-model threshold of 0, while `apply_model_thresholds` requires > 0 (CO-16). `request_timeout()` applies `max(1)` but has no upper bound (CO-03).
7. API stability: every serde key is a config-file key. `inflight` uses `skip_serializing_if`, which interacts with `pending` (CO-09). The untagged `DatasetRef` is fragile: an object with a typo becomes an error, not `Id`.
8. Tests: round-trip, legacy rename, names, device resolution and benchmark defaults are covered. `validate_config` on loaded files is not tested, because nothing calls it on that path.
9. Readability: well documented. The `Config` struct has 31 fields, and the README table lists only about 24 of them (CO-17).

### src/config_form.rs
1. Inputs: the untrusted form `HashMap<String,String>` (admin routes, loopback by default). Numbers go through `parse::<T>`, never unwrap. `port` and `request_timeout_secs` have lower bounds. Thresholds and nms_iou are checked to 0..=1 (NaN is rejected because `contains` is false for NaN). `worker_queue_size`, `gpu_index` and `intra_threads` have no upper bound (CO-03). `models_json` is parsed with serde, and per-model `inflight` is unbounded.
2. Failure paths: `on[i]` and `c.models[i]` use indices from `position`, so they are infallible.
3. Resources: clones the config, which is small.
4. Concurrency: none. It is called inside `ConfigStore::update` under the write lock.
5. Platform: `opt_path` takes paths as typed. Relative paths resolve against the exe dir later.
6. Correctness: the edit is atomic (works on a copy). The port error for "70000" says "is not a valid number" without the range. `cache_dir` and `log_path` accept anything (no existence or writability check).
7. API stability: form field names are shared with the config page JavaScript.
8. Tests: good coverage of rejection atomicity, markers, devices and selection. No test covers huge numbers.
9. Readability: good.

### src/config_merge.rs
1. Inputs: two `Config`s (already parsed).
2. Failure paths: `target.models[i]` at line 465 uses an index from `position`. `serde_json::to_value` failures fall back to an empty map, and `runtime_overlay` falls back to `current` on a deserialize error. Both fail silently but neither panics.
3. Resources: flattening clones the config twice per call. It runs per update and per `GET /v1/config`, which is fine at config sizes.
4. Concurrency: pure.
5. Platform: none.
6. Correctness: `live_global`, `pending` and `process` match README lines 206, 547 and 565 (port, bind_address and log_path need a process restart; allow_remote_admin applies at once). `pending()` iterates only the current model's keys (line 531), so a per-model field that is omitted when `None` (`inflight`) and was removed is never reported (CO-09). `model_reloads` compares full maps and does see it.
7. API stability: key syntax `model:<field>:<name>` is consumed by the UI.
8. Tests: classification is covered field by field. Removed optional fields are not tested.
9. Readability: good. `describe_changes` is long (70 lines) but readable.

### src/config_store.rs
1. Inputs: the config file on disk (edited by the CLI, editors or another process). It is reparsed with serde only, without `validate_config` (CO-04).
2. Failure paths: locks use `read_recover`/`write_recover`, so they are poison-safe. `as_millis() as u64` is fine. No unwrap outside tests.
3. Resources: the change log is bounded at 50 (`CHANGE_LOG_LEN`). Each record holds every changed field, so size is bounded by config size. File IO (read, write, read back) happens under the write lock, so concurrent `GET /v1/config` readers wait for a disk write.
4. Concurrency: one RwLock. `check_disk` takes a read lock, releases it, then takes the write lock; `sync_disk` re-stats under the write lock, so there is no lost update. Writes are not atomic (CO-02). The cheap pre-check skips the hash when mtime and length are unchanged (line 263), so a same-length edit within the same mtime tick is missed and later overwritten (CO-08).
5. Platform: mtime granularity is 100 ns on NTFS, 1 ns on APFS, but 1-2 s on FAT, some SMB/Docker bind mounts and HFS+ (CO-08).
6. Correctness: a failed `edit` leaves memory unchanged. A failed `save` partway through leaves memory unchanged but the file truncated (CO-02). `adopt` bumps the generation even when `config` is None, as documented.
7. API stability: `ChangeRecord` is camelCase JSON for the UI.
8. Tests: bumps, external change, broken file and adopt are covered. Coarse mtime and write failure are not tested.
9. Readability: good docs.

### src/runner.rs
1. Inputs: the config file on restart (`reload_config`: load without validation, falls back to current on error or empty models).
2. Failure paths: `expect("listeners are used once")` at line 279 and `expect("server task is running")` at line 301 are both infallible by construction (serve runs before serve_until_event, and the task is cleared before returning).
3. Resources: the watcher thread is joined in `retire`. `retire` sleep-polls up to DRAIN (5 s) on the main thread. `finish` calls `block_on(server task)` with no timeout (axum graceful shutdown waits for open requests) (CO-12). `benchmark.wait_idle(30 s)` on restart.
4. Concurrency: the 2-worker multi-thread runtime handles HTTP. Restart builds the new generation while the old one serves. Generation, workers and server_stop are all child tokens of `shutdown`, so a signal reaches every one. `resources_restart` uses SeqCst.
5. Platform: signals are handled by the binaries, not here.
6. Correctness: the restart loop is sound. A failed restart restores `previous`, and if the restore also fails, the process exits even though the old generation was still serving (rare: `start_registry` only fails on empty models or duplicate names, which the restored config passed before). `ensure_models` runs before `bind` (line 351 vs 355), so with no models there is no UI at all (CO-05).
7. API stability: none.
8. Tests: the generation and restart loop is exercised by server integration tests (not reviewed here). There is no unit test for restore-on-failure.
9. Readability: good module docs. `run_server_with` is 135 lines and dense but commented.

### src/startup.rs
1. Inputs: none.
2. Failure paths: raw `RwLock`. A poisoned lock makes `get` return `Failed("poisoned")` and makes `set` a silent no-op. The critical sections only clone or assign strings, so poisoning is practically impossible. It is still inconsistent with `crate::sync::*_recover`.
3. Resources: one Arc per model.
4. Concurrency: `describe` takes the lock twice (`get`, then `detail`), so it can mix two states. Harmless: the detail is filtered on Initializing.
5. Platform: none.
6. Correctness: the state machine has no guarded transitions (any to any). That is intended, because reloads go Ready to Initializing.
7. API stability: the `describe` strings are shown in the UI and in "Model 'x' is initializing" answers.
8. Tests: one test, which is sufficient.
9. Readability: good.

### src/system_info.rs
1. Inputs: `/proc/cpuinfo`, `/proc/meminfo`, `sysctl` output (absolute `/usr/sbin/sysctl`), and PowerShell output on Windows.
2. Failure paths: `kb * 1024` is unchecked, but MemTotal would need more than 2^54 kB to overflow, so it is infallible in practice. All parse failures give `None`.
3. Resources: spawns processes (macOS sysctl, Windows `powershell`). Neither has a timeout (CO-13). Called once per process for the profile (OnceLock in `profile.rs`) plus once in the startup log line.
4. Concurrency: none.
5. Platform: Windows resolves `powershell` by search path, not `%SystemRoot%\System32\WindowsPowerShell\v1.0\powershell.exe`. PowerShell start-up costs about 0.5-2 s. On Linux non-x86 the "Model" and "Hardware" keys are used. x86 uses cpuid.
6. Correctness: the memory value feeds the hardware profile (low/mid/high). `None` changes the class silently (see profile.rs, not reviewed here).
7. API stability: none.
8. Tests: smoke test only.
9. Readability: fine.

### assets/coco_classes.yaml (asset)
- Format: `NAMES:` list of 80 entries, ASCII, LF line endings. It is embedded with `include_str!` and parsed with an `expect` in `model/classes.rs:65-66`, which is safe because the bundled content is fixed.
- Content: Darknet spellings (`motorbike`, `aeroplane`, `sofa`, `pottedplant`, `diningtable`, `tvmonitor`). Ultralytics-exported models with their own `metadata.yaml` use `motorcycle`, `airplane`, `couch` and so on, so the label a Blue Iris user filters on can differ by model. Whether CodeProject.AI's YOLOv5 module uses the same spellings was not checked here (unverified); no finding.
- License and provenance: the COCO category names, no license concern.

### From the non-Rust+tests (Opus subagent) report

The Rust test files get the nine Rust items, kept short. Every other file gets the non-Rust rubric.

### tests/common/mod.rs
1. Inputs: env `BLUE_ONYX_TEST_ROOT`, fixture files. A missing required fixture is a hard `assert!` failure, never a silent pass. Good.
2. Failure paths: `expect`/`assert` everywhere. That is fine in tests. `free_port` has a TOCTOU race, which is handled by the 5-attempt retry in `Server::start` (line 263).
3. Resources: `temp_dir` (line 119) creates `bo_test_*` folders that are never removed, so they pile up in the system temp dir (NR-24). `Server::stop` waits at most 30 s for the registry to be released and 30 s for the workers to stop.
4. Concurrency: the harness uses one global log handle (`OnceLock`) and per-test runtimes. `Runner` mode runs the production `run_server_on` on its own thread, as the service does.
5. Platform: portable.
6. Correctness: the doc comment matches the behavior (ignored + hard failure).
7. API: it drives `/v1/vision/*`, `/config*`, `/stats.json`, `/prometheus`, `/v1/config`.
8. Tests: this is the shared harness.
9. Readability: good.

### tests/backend_cpu.rs
1-5. Fixture driven; asserts the IPcam shapes `[1,25200,8]`, the CPU spec, the length check and `infer_into` buffer reuse.
6. Line 110 is a stale comment ("ONNX Runtime is not in this build yet"). ORT is a default feature. The assertion still holds only because `explicit_candidates` (src/backend/plan.rs:183) falls back to OpenVINO CPU when no ORT library or CUDA is present (NR-23).
7-9. Covers GPU->CPU fallback flags when no GPU is listed. It cannot assert a real GPU on CI.

### tests/backend_ort.rs
1-4. Uses a process-wide `Runtimes` behind a `Mutex`. The cache dir is the fixed `temp/bop-ort-test-cache` (line 58), shared across runs (NR-24).
5. CoreML parts are `cfg(target_os="macos")`. CPU on every OS.
6. `yolo5_on_coreml` asserts exactly one rewritten copy in that shared cache (line 244). A stale copy from an earlier `REWRITER_VERSION` or model sha makes it fail on a dev machine.
7. Asserts the provider strings "ONNX Runtime CPU" and "ONNX Runtime CoreML".
8. The RT-DETR i64 input and the CoreML pinned-batch guard run only when `rt-detrv2-s.onnx` exists. CI never downloads it (NR-14).
9. OK.

### tests/integration_http.rs
1-9. Covers the default route, the custom route, an unknown model, and the list, on the IPcam model on CPU. Clean.

### tests/families_http.rs
1-9. Only IPcam is required. YOLO26 IR (explicit family and `auto`) and RT-DETR are optional and absent in CI, so in CI this test duplicates integration_http (NR-14).

### tests/enabled_models_http.rs
1-9. Covers disabled models, zero enabled models, the Models card save (live) vs restart, requests during a restart (no connection errors), and process-lifetime counters. Without `rt-detrv2-s.onnx` the second model is an IPcam copy, so the COCO label check is skipped (NR-14).

### tests/live_config.rs
1-9. Covers the threshold hot-apply, disable/enable without restart, and a per-model device reload (generation unchanged, `pendingCount` 0, change log source). Runs on any platform with `auto_download: false`. Clean.

### tests/multi_model_http.rs
1-9. Covers concurrent requests to two models, case-insensitive names with extension, exact counters, and Prometheus series. The lazy-model part (initializing answer < 2 s, then success) runs only with `yolo26n.xml`, which is absent in CI (NR-14). Lazy logic has unit coverage in `src/worker.rs` / `src/server/detect.rs` (grep `trigger_load`, `lazy: true`).

### tests/select.rs
1-9. Hermetic and pure. It covers the iGPU-only primary target, NVIDIA+iGPU, Pascal (CUDA skipped on Windows), AMD, largest VRAM first, Mac arm64/x86, CPU-only/none, Arc + iGPU + NPU (never auto), `.xml`-only models, serialization, and the measured order. Good coverage of the auto ranking. Clean.

### tests/data/dog_bike_car.jpg
Binary test image, also embedded in binaries per `src/bin/test_blue_onyx_prism.rs` ("embedded dog_bike_car.jpg"). Its provenance and license are not documented in LICENSE.

### scripts/export_model.py
- Pinned deps: in service mode it runs inside the hashed lock env, with `python -u -I` and the offline env vars (`YOLO_OFFLINE`, `HF_HUB_OFFLINE`, `TRANSFORMERS_OFFLINE`, `YOLO_AUTOINSTALL=false`), per export.rs:1153-1196. By hand, `requirements-export.txt` uses ranges.
- shell=True: none. No subprocess at all.
- Paths: `name` goes into `args.out / f"{name}.onnx"` (lines 360, 369, 684, 300, 309) without normalization. In service mode `name` comes from the catalog. By hand, `--name ../x` would write outside `--out` (user's own arg; low).
- Determinism: fixed synthetic image. The COREML_CACHE_KEY is the sha256 of the serialized graph (line 280-290).
- `--help`: the docstring is accurate. One gap: `--int8 --data coco128.yaml` makes Ultralytics download the calibration set, which is undocumented network use (by-hand only).
- Exit codes: `fail()` raises `SystemExit(msg)`, so exit 1 with the message on stderr. Uncaught exceptions also exit 1. export.rs treats non-zero as failure (`!status.success()`, export.rs:662).
- Cross-check with src/resources/export.rs:
  - Arguments `--recipe --name --family --weights --imgsz --no-openvino --out-dir --work [--source]` all exist in `parse_args` (lines 116-147, `--out-dir` with alias `--out`).
  - Output names `<name>.onnx` / `<name>.yaml` in `--out-dir` match `install_outputs(staging.join(onnx_name))` (export.rs:1243).
  - The 80-class YAML matches `class_count == 80` (export.rs:1252).
  - `stamp_coreml_cache_key` is applied to every `.onnx` written (lines 371, 686).
  - Mismatch with the docs: the Ultralytics recipe never runs `compare_with_onnxruntime`. Only the RF-DETR and DEIMv2 recipes do (NR-08).
- The Ultralytics path loads the model three times (`ultralytics_model` at lines 344, 365, 373) and copies the `.pt` into the work dir each time if needed. This wastes time but is harmless.

### scripts/convert_onnx_to_ir.py
By-hand tool; network: none; exit non-zero via SystemExit / exception; `--input-shape` parse errors raise a traceback (non-zero). Copies the sibling YAML. Clean.

### scripts/build_bench_sets.py
- Network: documented (HF, COCO S3). Downloads are SHA-checked where a hash is known (COCO zip pinned, HF LFS oids). File names go through `safe_name` (line 116). Zip members are read by name. Nothing is executed.
- Determinism: candidates are sorted, then shuffled with a seeded RNG. Day/night uses Pillow luma, but `Pillow>=10` is unpinned (NR-26).
- Validation: `validate()` relies on `assert` (lines 688-703), which `python -O` strips (NR-26).
- Exit codes: non-zero on errors.
- No shell.

### scripts/ci/smoke.sh
- `set -euo pipefail` (line 10). Variables are quoted. No `curl | sh`.
- A `success:false` response does fail the script: line 54 `grep -q '"success":true'` under `set -e`.
- A wrong provider fails it: lines 55-57.
- Zero predictions pass, though, and `$EXPECTED` is used as an unescaped ERE (NR-15).
- The cleanup trap kills the server and prints its log.
- Startup waits are unbounded in effect: up to 120 s + 120 s even if the server died at once.

### scripts/install_systemd.sh
- `set -euo pipefail`. Root is needed but not checked: there is no `id -u` test, and the first privileged command fails with a less clear error.
- It creates the system user `blueonyx` and gives that user ownership of the whole install dir, binary and libraries included (NR-04).
- `sed "s#...#$INSTALL_DIR#g"` breaks on a `#` or `&` in the path, and a relative `INSTALL_DIR` produces a broken unit.
- Partly idempotent: on a rerun `enable --now` does not restart a running service, so an upgraded binary is not picked up (NR-21).
- No uninstall script exists.
- The file is mode 100644 in git. That is fine in the release archive (chmod in release.yml:84), but it fails from a checkout.

### scripts/install_launchd.sh
- `set -euo pipefail`; root needed (writes /Library/LaunchDaemons) but not checked.
- The plist has no `UserName`, so the daemon runs as root (NR-03).
- The comment on line 2, "CPU inference only on Apple silicon", is stale: CoreML is used via ORT.
- Idempotent (bootout then bootstrap). No uninstall script.

### scripts/install_service.ps1
- `#Requires -RunAsAdministrator` and `$ErrorActionPreference="Stop"` are present. Stop does not cover native `sc.exe` exit codes, which are never checked (NR-16).
- The service runs as LocalSystem by default (NR-02).
- `-Account` passes the password on the `sc.exe` command line (line 71, NR-16).
- It changes the machine-wide `ServicesPipeTimeout`, and uninstall never reverts it (NR-21).
- The firewall rule covers all profiles and any remote address.
- Idempotent by delete+recreate, subject to the 1072 race (NR-16).

### scripts/uninstall_service.ps1
`$ErrorActionPreference="Continue"`, best effort. Removes the service, the firewall rule and the event source. Does not restore `ServicesPipeTimeout`, and does not touch the install dir or config. Acceptable (NR-21).

### scripts/model-export/gen_locks.sh
- `set -eu` (POSIX sh, no pipefail needed since there are no pipelines that matter). Temp dir cleaned by an EXIT trap. `--exclude-newer` freezes transitive deps.
- `cmp` asserts that the three CPU-index resolutions (linux x64, linux arm64, win x64) are identical. That makes one lock valid for all three.
- `$1` is unchecked: `set -u` aborts with "parameter not set". Acceptable.

### scripts/model-export/locks/*.txt
Every requirement is `pkg==ver` followed by one or more `--hash=sha256:` lines (63/63 in each). There are no option lines (`--index-url`) inside the locks. The CPU index is supplied by `uv --torch-backend cpu`.

### scripts/requirements-*.txt
`requirements-model-export.txt` uses exact pins and is the input to the locks; export.rs:2375 tests that the locks contain every pin. `requirements-export.txt` uses ranges (by hand, documented). `requirements-bench.txt` uses `Pillow>=10` (NR-26).

### .github/workflows/ci.yml
- All actions are pinned by SHA. `--locked` is on every cargo invocation.
- There is no `permissions:` block (NR-05) and no `pull_request_target`. No secrets are used.
- The cache is Swatinem/rust-cache with default keys.
- `continue-on-error` is on the two DirectML steps (lines 95, 99). This is a known gap, documented in the workflow comment and RELEASE_CHECKLIST step 1.
- Ignored tests run with `--ignored --test-threads=1` against `target/release`, but only with `ipcam-bird` + `IPcam-general` (NR-14).
- No `timeout-minutes` (NR-20).

### .github/workflows/release.yml
- Actions are pinned. `--locked` is on the build.
- `permissions: contents: write` is set at workflow level, so the build jobs get write access too (NR-05).
- The tag must equal the Cargo version (line 56).
- `workflow_dispatch` creates or updates a release for `tag_name = VERSION` (line 144). It can touch an existing published release (NR-07).
- The rust-cache is restored in release builds (cache scoped to tag/main refs).
- The archive ships OpenVINO/ORT/DirectML binaries without their licenses (NR-17).
- No signing or notarization. The `.sha256` is generated in the same job (integrity only, not authenticity).

### .github/workflows/check-urls.yml
- `permissions: contents: read`. Actions pinned.
- `cargo run` has no `--locked` (line 25).
- A failure only shows as a red scheduled run: no issue, no notification other than GitHub's default email to the cron's last editor. Scheduled workflows are disabled after 60 days without repo activity (NR-20).

### .github/dependabot.yml
cargo + github-actions, weekly, grouped minor/patch. There is no `docker` ecosystem. That is intentional: the base-image tag must equal `OPENVINO_VERSION`. No pip, also intentional (locks are regenerated by hand). Clean.

### Cargo.toml
- `ort` (`=2.0.0-rc.13`) and `openvino-sys` (`=0.11.0`) are exactly pinned with comments.
- Unused direct deps: `imageproc`, `mime`, dev-dep `approx`. The clap `env` feature is unused (NR-22).
- Release profile: `opt-level=3, lto="thin", codegen-units=1, strip=true`, panic = unwind (default). Unwind is required: `catch_unwind` in src/prep.rs:304 and src/benchmark/service.rs:379.
- `strip=true` with no split debuginfo leaves crash backtraces in the field without symbols. Consider `debug="line-tables-only"` + `split-debuginfo` uploaded as a release artifact.

### deny.toml
- Advisories: two ignores with a reason, but with no date or review-by (NR-25).
- Licenses: an explicit allow list. `cargo deny --offline check bans licenses sources` passes ("bans ok, licenses ok, sources ok"), with 2 allow entries never encountered (`BSD-2-Clause`, `MPL-2.0`).
- Bans: wildcards denied, duplicates warn.
- Sources: crates.io only.

### rust-toolchain.toml
Pins 1.99.0 but lists no `components` (rustfmt, clippy). The CI action installs them. Local agents hit this per the user's memory note (NR-25).

### .gitignore / .gitattributes / .dockerignore
`.gitignore` covers weights (`*.pt`), models, cache, runtimes, configs (`*_config*.json`; `deploy/docker-config.json` is not matched because it uses a hyphen), `.venv`, `research_notes/`, `reports/`. `.gitattributes` forces LF (CRLF for ps1) and marks binaries. `.dockerignore` excludes target/.git/models/cache/openvino/.venv/configs but not `onnxruntime/`, `tools/` or `bench/`, which can bloat the build context. Clean.

### Dockerfile
- Least privilege: runs as `openvino` (non-root).
- The base tag `openvino/ubuntu24_runtime:2026.4.0` equals `OPENVINO_VERSION` "2026.4.0" (catalog.rs:42).
- No HEALTHCHECK, although `curl` is installed (line 20) and never used (NR-18).
- The builder is the floating `rust:1-bookworm` (line 9). rust-toolchain.toml then makes rustup download 1.99.0 at build time.
- Build-time network: `setup-onnxruntime` and `fetch --for-config` download hash-pinned resources. OK.
- Volumes: the config defaults are consistent with docker-config.json (`/app/models`, `/app/cache`).
- The entrypoint copies the default config on first start (NR-19).

### deploy/docker-compose.yml / deploy/docker-config.json
- The port is published on all host interfaces, and `allow_remote_admin: true` (NR-01).
- `models` is a named volume, which contradicts the README "Put models in deploy/models" (NR-09).
- `devices: /dev/dri` makes compose fail on hosts without it (no fallback documented).
- docker-config keys (`port, allow_remote_admin, device, auto_download, allow_large_downloads, openvino_dir, onnxruntime_dir, models_dir, cache_dir, models`) all exist in `Config` (src/config.rs:177-242).

### deploy/blue-onyx-prism.service
- `User=blueonyx`, `SupplementaryGroups=render video`.
- No hardening: `NoNewPrivileges`, `ProtectSystem=strict` + `ReadWritePaths`, `PrivateTmp`, `ProtectHome` are all missing (NR-04).
- `After=network.target` is fine for `0.0.0.0` (a specific `bind_address` needs `network-online.target`).

### deploy/com.blueonyx.prism.plist
Runs as root (no `UserName`), with `KeepAlive`. stdout and stderr both go to one log file with no rotation (NR-03, NR-06).

### README.md
- Commands checked against src/cli.rs, src/benchmark/cli.rs and src/bin/test_blue_onyx_prism.rs:
  - Subcommands `setup-openvino [--dest --version --archive]`, `setup-onnxruntime --flavor auto|cpu|cuda|directml --dir`, `download-models --name/--all/--dir/--add-to-config`, `add-model`, `list-models`, `fetch`, `list-resources --check-urls`, `list-devices --model` all exist with those flags.
  - Benchmark flags (`--all-devices --apply --apply-threshold --threshold-objective --threshold-search --tag --class --search-model --search-device --iou --report --no-save --force-cpu --image --dataset --images --gt --max-images --list-datasets --reference-model --accuracy-weight --repeat --warmup --compare-cpu --cache-dir --threads --classes --min-confidence --config --json -v`) exist in BenchArgs.
  - Test-client flags all exist, and it does exit non-zero on failure (`bail!` at line 234).
- Config table: every key listed exists with the stated default (src/config.rs:342-376). Missing from the table: `preprocess_threads`, `inflight`, `ort_sessions`, `hardware_profile` and the model-level `inflight` (only the `benchmark` section is shown elsewhere).
- Stale statements: NR-10, NR-11. Docker: NR-09.
- License attribution for blue-onyx (MIT, Marcus Asteborg), Ultralytics (AGPL-3.0), RF-DETR (Apache-2.0), DEIMv2 (non-commercial) and DINOv3 is present. Third-party runtime licenses are missing (NR-17).

### CLAUDE.md
- Commands checked as for the README; all exist.
- Line 38 claims every export is checked against PyTorch with ORT (NR-08).
- Lines 116-117 say "HTTP integration test skips when `models/` is absent". It is `#[ignore]` and hard-fails without fixtures (NR-23).

### docs/BENCH_IMAGES.md
Numbers and the schema match `assets/bench/*.json` and the script (complexity rules, luma threshold 50, seed 20261010). Clean.

### docs/PLAN.md
Historical banner on line 1. Clean.

### docs/RELEASE_CHECKLIST.md
- The CUDA step uses a GPU the pinned Windows CUDA build cannot run (NR-12).
- There is no step for the primary target's Intel iGPU (NR-13).

### docs/REVIEW_PLAN.md
Historical; decisions respected. Line 369 is the source of the GTX 1070 Ti step (NR-12).

### LICENSE
- MIT plus the blue-onyx attribution, plus attribution for the benchmark datasets.
- Missing: Roboto Mono (Apache-2.0, embedded in the binary), and OpenVINO (Apache-2.0), ONNX Runtime (MIT + ThirdPartyNotices) and DirectML (Microsoft license) shipped in release archives (NR-17).

### .claude/agents/engineer.md, porter.md
`porter.md:14-15` says "never copy ONNX Runtime specific code". That contradicts CLAUDE.md "Copy ideas ... including its ONNX Runtime usage" (NR-23).

### assets/favicon.ico, assets/roboto-mono-stripped.ttf
Binary; noted only. The font is Apache-2.0 (src/image.rs:4).


## Resource and concurrency notes per area (incl. the full route table)

### From the backend (Fable subagent) report

- Listeners: none in these files.
- Filesystem writes: OpenVINO `cache_dir` (created in `apply_core_properties`, device.rs:131); ORT `cache/tensorrt`, `cache/coreml` (ort.rs:645-652); CoreML rewritten models in `cache/coreml/<stem>-<sha16>-v<N>.onnx` or, with no `cache_dir`, `<tmp>/blue-onyx-prism-coreml/` (ort.rs:606, BE-04), written as `.<name>.tmp-<pid>-<seq>` then renamed (onnx_rewrite.rs:1240-1256); `onnxruntime/active.txt` via `.tmp` + rename (libs.rs:574-586); `PATH`/`OPENVINO_INSTALL_DIR` process environment on Windows.
- Downloaded content parsed/executed: every shared library under `openvino/`, `onnxruntime/<flavor>/`, `onnxruntime/cuda-libs/` is `dlopen`ed (initializers run) — by design, after SHA-256 verification in the manager; the `.onnx` model file is parsed at the protobuf wire level only on the CoreML path (onnx_rewrite.rs) and otherwise handed to the runtime.
- External programs: none run from these files.
- Threads/channels: OpenVINO: `n = tuning.effective_inflight(streams)` infer requests per worker, one unbounded completion channel (at most `n` outstanding). ORT: one OS thread per session (`ort-session-<slot>`), `bounded(1)` job and result channels per slot, one unbounded completion channel. `Drop` joins the threads after dropping the job senders; a thread blocked inside `Session::run` cannot be interrupted (documented).
- Process-wide singletons: `detect::hardware()` (OnceLock), `libs::PRELOADED`, `libs::CUDA_PRELOADED`, `libs::LOADED_OPENVINO_C`, `libs::NOTES`, `ort::LOADED`, `device::CPU_DEFAULT_PRECISION`. All use non-panicking lock access.
- Hot path per inference (OpenVINO): copy CHW into the slot tensor, optional extra `Tensor` allocation (RT-DETR), per-output `get_output_tensor_by_index` + `get_shape` wrapper allocations, output copy into recycled `Vec`s. (ORT): CHW copy into the slot `Vec`, `extra.to_vec()`, two `String` clones, `Vec<SessionInputValue>`, ORT's own output allocation, copy into recycled `Vec`s. No per-request allocation grows unboundedly (verified `OutputBuf::f32_mut` clears while keeping capacity).

### From the pipeline (Fable subagent) report

- Listeners: none here (server.rs). Filesystem writes: `image::save_annotated` (`<save_image_path>/<stem>_od.jpg`, `<stem>.jpg`), created with `create_dir_all`.
- Downloaded content parsed: none in these files (class YAML and `preprocessor_config.json` are local files). External programs: none.
- Threads: one `worker-<name>` per enabled model (registry); `prep-<i>` pool (`preprocess_threads`, >= 1, <= 32); `stop-<name>` helper per retired worker. Channels: work `bounded(queue_size | 64)`; prep queue `bounded(threads*16)`; prep reply `bounded(1)` per request; backend completions (backend-owned). Tick every 250 ms when idle.
- Lock order: `Spawner.reload` -> `slots` (write) ; `runtimes` is never taken while `slots` is held. `runtimes` is held for a whole compile by the loading worker; all UI readers `try_lock`, only `plan.rs:49` and `registry.rs:629` block.
- Hot path per request (steady state): server `Bytes` (0-copy into `PrepJob`) -> `Meta` + `bounded(1)` channel (2 small allocs) -> prep thread: `RgbImage` decode buffer (w*h*3, **allocated per request, 6 MB at 1080p**; `image` fallback allocates twice) -> letterbox into a pooled CHW tensor (reused) -> `extra_inputs` Vec (RT-DETR only) -> backend copies the tensor into the infer request; outputs refill reused `NamedOutput` buffers -> `dets` Vec + `best`/`best_cls` (yolo8) -> `to_predictions` (one `String` per prediction) -> response strings (`provider.clone()`, message) -> JSON in the server. Image saving (clone + JPEG encode + write) runs on the worker thread when enabled.
- Reload: the old worker leaves routing immediately; its engine is dropped when its thread exits (drain <= request_timeout + 1 s, then cancel). No handle leak found; two compiled models coexist transiently.
- Poisoned mutexes: all `lock_recover`/`read_recover` except `worker.rs:620-625, 1123` (silently skip).

### From the resources (Fable subagent) report

- Listeners: none in these files (tests start throw-away `TcpListener`s on 127.0.0.1).
- Filesystem write paths: `<root>/.downloads.lock`, `<root>/.downloads/<file>[.partial]`, `<root>/<dest>` via
  `<parent>/.<name>.staging` and `.<name>.old-<pid>` (manager.rs 904-990), `<models>/<file>.partial` → `<file>`,
  `.installed.json` / `.<model>.installed.json` (+ `.tmp`), `onnxruntime/active.txt`, `VERSION`, `flavor.txt`;
  export: `<root>/tools/{uv,export-env,python,uv-cache,export-weights/<name>,export-work/{export_model.py,<name>},
  export-config,deimv2-src,.model-export.lock,model-export-requirements.lock.txt}`, `<models>/.<name>.export-staging`.
  Everything is under the data root or the configured models dir; the only user-chosen targets are `setup-* --dest/--dir`.
- Downloaded content parsed/extracted: `extract.rs` (zip via `zip` 9 with central-directory filtering, tar.gz via
  `tar`/`flate2`) after SHA-256; `.installed.json` (serde, lenient); GitHub release JSON (`update.rs`); the export
  script parses `config.json`/safetensors/`.pt` inside the Python child (torch `pickle` for `.pt` – hash-pinned).
- External programs: only `export.rs::SystemRunner` runs `tools/uv/uv` and `tools/export-env/bin/python` (never
  anything from PATH); arguments are built from static strings and our own paths; stdin null, stdout/stderr piped.
  Not in a process group / Job Object (RS-02). Env: inherited minus a prefix blocklist plus explicit offline vars (RS-07).
- Threads/channels: `downloads` thread (1, unbounded `Msg` channel, dedup by key), `model-export` thread (1, unbounded,
  dedup by id), 2 reader threads per child (unbounded line channel, drained every 100 ms), unbounded subscriber channels
  (`Manager::subscribe`). Condvars for `wait`. The detection route never touches these; `Wait::poll` is called by model
  workers while Initializing only.
- Hot path: none of these files are on the detection request path. `GET /v1/resources` (`status::rows`) is fs-heavy
  (RS-14) and must stay on `server::blocking`.

### From the server+UI (Fable subagent) report

### Route table (src/server.rs:132-192)

Guard column: **open** = reachable by any client; **admin** = `admin_only` middleware (loopback peer,
or anyone when `allow_remote_admin`). State = whether the handler changes server state. Blocking =
whether heavy work runs inside `server::blocking` (B), on the async thread (A), or is trivial (-).

| Method | Path | Handler | Guard | State-changing | Blocking |
|---|---|---|---|---|---|
| GET | `/` | `pages::welcome` | open | no | B |
| POST | `/v1/vision/detection` | `detect::detection_default` | open | no (counters) | - (async) |
| GET, POST | `/v1/vision/custom/list` | `detect::custom_list` | open | no | - |
| POST | `/v1/vision/custom/{model}` | `detect::detection_custom` | open | no (counters, lazy load trigger) | - |
| GET | `/v1/status/updateavailable` | `pages::update_available` | open | no (outbound GitHub, cached 1 h) | - |
| GET | `/v1/devices` | `devices::devices_json` | open | no | B |
| GET | `/v1/config` | `live::config_json` | **open** | yes-ish: `sync_config` applies an on-disk change (reloads models) | B |
| GET | `/stats.json` | `pages::stats_json` | open | no | - |
| GET | `/prometheus` | `pages::prometheus` | open | no | - |
| GET | `/static/style.css`, `/static/live.js`, `/favicon.ico` | static | open | no | - |
| GET | `/v1/resources` | `resources::resources_json` | admin | no | B |
| POST | `/v1/resources/download` | `resources_download` | admin | yes (starts downloads) | B |
| POST | `/v1/resources/remove` | `resources_remove` | admin | yes (deletes files) | B |
| POST | `/v1/resources/add-to-config` | `resources_add_to_config` | admin | yes (config write + worker) | B |
| POST | `/v1/models/add` | `models_add` | admin | yes (config write + worker) | B |
| GET | `/v1/fs/browse` | `browse::browse` | admin | no (lists any directory) | **A** |
| POST | `/v1/resources/export` | `resources_export` | admin | yes (downloads + runs the export toolchain) | B |
| POST | `/v1/resources/export/cancel` | `resources_export_cancel` | admin | yes | B |
| GET | `/stats` | `pages::stats_page` | admin | no | - |
| GET, POST | `/test` | `test_page` / `test_submit` | admin | no (runs a detection) | async + spawn_blocking for drawing |
| GET, POST | `/config` | `config_page` / `config_submit` | admin | POST: yes (config file + live apply) | B |
| POST | `/config/models` | `config_models` | admin | yes (+ optional restart) | B |
| POST | `/config/restart` | `config_restart` | admin | yes (restarts the generation) | - |
| POST | `/config/loglevel` | `config_loglevel` | admin | yes (log level + config file) | B |
| GET | `/logs` | `logs_page` | admin | no | - |
| GET | `/logs.json` | `logs_json` | admin | no | - |
| GET | `/benchmark` | `benchmark::page` | admin | no | B |
| GET, POST | `/v1/benchmark` | `status` / `start` | admin | POST: yes (starts a run) | GET: B, POST: **A** |
| POST | `/v1/benchmark/cancel` | `cancel` | admin | yes | - |
| POST | `/v1/benchmark/apply` | `apply` | admin | yes (config write + reload/restart) | **A** |
| POST | `/v1/benchmark/apply-threshold` | `apply_threshold` | admin | yes (config write + live apply) | **A** |
| GET, POST | `/v1/benchmark/threshold-search` | `search_options` / `threshold_search` | admin | no | **A** (multi-MB JSON parse + search) |
| POST | `/v1/benchmark/settings` | `settings` | admin | yes (config write) | **A** |
| GET | `/v1/benchmark/images` | `images_detail` | admin | no | **A** (results JSON parse) |
| GET | `/v1/benchmark/image` | `image_file` | admin | no | **A** (results JSON parse per image) |
| any | anything else | `pages::fallback` | open | no | - |

Findings from the table: every state-changing route is behind the loopback/`allow_remote_admin`
guard, and the Blue Iris routes plus `custom/list` are correctly open. `/v1/config` is the one open
route that leaks configuration (SV-05) and can trigger a model reload. All admin POSTs are plain
urlencoded/multipart form posts with no CSRF token, Origin/Sec-Fetch-Site or Host check (SV-02).
Unknown methods on known paths get axum's 405 with an empty body; unknown paths get 404 JSON echoing
the path (escaped).

### Listener / HTTP
- `bind` (server.rs:308-330): IPv4 listener + best-effort `[::1]` listener; bound once per process.
- `serve_on` (334-356): `axum::serve(..).with_graceful_shutdown`. No `header_read_timeout` timer
  (axum 0.8.9 does not install a hyper `Timer`, so hyper's 30 s default is inert — unverified at
  runtime), no idle keep-alive timeout, no connection/in-flight limit, no per-request timeout layer
  (SV-04). Body limit 32 MiB (`DefaultBodyLimit`), hyper request-head buffer ~408 KiB.
- Handler panics are not caught (no `CatchPanicLayer`); the connection is closed (SV-15).

### Filesystem write paths reachable from these handlers
- Config file (`config_store::update` -> `Config::save` = plain `std::fs::write`, not atomic; owner:
  config_store) from `/config`, `/config/models`, `/config/loglevel`, `/v1/models/add`,
  `/v1/resources/add-to-config`, `/v1/benchmark/apply`, `/apply-threshold`, `/settings`, export `OnDone`.
- `save_image_path` / `log_path` / `cache_dir` set verbatim from the UI (SV-03); the actual writes
  live in the worker/cli (not reviewed here).
- Resource downloads/removals/exports under the data root (resources module).
- `benchmark.json` / `benchmark-preds.json` next to the config (benchmark service).

### Downloaded content parsed / external programs
- `/v1/status/updateavailable` parses GitHub release JSON (reqwest, 10 s timeout, cached).
- `/v1/resources/export` starts the export pipeline (`resources/export.rs`): the only code that runs
  downloaded programs; user-triggered, but reachable cross-site (SV-02).

### Threads / channels
- Per model: one worker thread + bounded crossbeam channel (`worker_queue_size` / soft capacity);
  `reload_model` spawns a replacement worker and a `stop-<name>` helper thread (joined lazily).
- `blocking()` uses tokio's blocking pool (default cap 512 threads).
- Static caches: `BENCH_CACHE`, `SAVED_CACHE` (one entry each), `update::CACHE` (one entry).
- Log ring: 2000 entries, unbounded bytes per entry (SV-08).

### Hot-path allocations per detection
Multipart parse (image `Bytes` copy, `HashMap` of text fields), one `oneshot`, response struct;
`command.to_string()`; `handle.name` clones only on error paths. Fine.

### From the benchmark-core (Opus subagent) report

- **Filesystem writes**:
  - `benchmark.json` beside the config (`BenchmarkResults::save`: tmp `benchmark.json.<pid>.tmp` plus rename; tmp is left on error). The CLI writes it via `save_results` (cli.rs:1015) and the service via `ServiceRun::save` (service.rs:947).
  - `benchmark-preds.json` via `search::save_beside` (out of scope).
  - The config file via `--apply` / `--apply-threshold` (`Config::save`).
  - `--report <path>` via a plain `std::fs::write` (export.rs:878).
  - Compiled-model cache dir (`cache_dir`) is only listed (`cache_files`) to detect a hit or miss.
- **Parsed external content**: `benchmark.json` (serde_json, error not panic), dataset images (decoder), model files (backends). No archives are extracted and no programs run in these files. Dataset downloads are delegated to `resources::actions::download` and `resources::commands::fetch`.
- **Threads/channels**: one detached `benchmark` thread per service run (no JoinHandle). Progress is published through `Mutex<Inner>`. Cancel is an `AtomicBool` checked per iteration, per device and per download poll (250 ms). Compile and model load are not interruptible.
- **Locks**: `Inner` (service) and `Mutex<Runtimes>` (shared with registry workers) never nest. The runtimes lock is held only for plan, selection and compile. Benchmark inference on shared runtimes runs concurrently with production inference on the same device, which affects production latency during a run (by design).
- **Memory**: the service keeps full per-image results for every finished model until the final save (BM-11). `Status` holds summaries only. `sweep` clones all finished device results after each device. Eval preds at 0.05 are uncapped per image.
- **Cross-file observations (other reviewers' files)**: `server/benchmark.rs:756` loads and parses the whole `benchmark.json` (sync `std::fs` + serde) on every `/v1/benchmark/image` request inside an async handler. `server/live.rs:100` also calls `load_or_warn`.

### From the benchmark-metrics (Opus subagent) report

- No listeners, threads or channels in these files.
- Filesystem writes: `StoredPreds::save` (search.rs:189-197) writes `benchmark-preds.json.<pid>.tmp` next to the config and renames it; `create_dir_all` error ignored; tmp leaked on rename failure.
- Downloaded content parsed: the built-in set images (downloaded per catalog, SHA-256 checked by resources) are only header-read here (`image_dimensions`); the manifests themselves are embedded, not downloaded. User-dir `manifest.json`, COCO JSON, YOLO txt/yaml are parsed with serde/serde_yaml_ng, unbounded size.
- No external programs.
- Memory per prediction set: `RunResult.eval_preds` (not in these files) holds every 0.05-threshold prediction with a `String` label; `StoredRun` holds the scored-class ones for every (model, device) ever kept; nothing caps detections per image except the model's own output size (e.g. DETR top-300). The preds file is loaded whole for each search request (BQ-04).
- Cross-file (not mine, for the server/benchmark-core reviewers): `threshold_search` and `search_options` (src/server/benchmark.rs:1007, 1029) do file IO + `search()` directly on the 2-thread tokio runtime; `percentile` (src/benchmark/mod.rs:109-112) panics on an empty slice (`clamp(1, 0)`).

### From the core/cli/config (Opus subagent) report

- **Listeners:** bound once per process in `runner::run_server_with` (runner.rs:355), after `ensure_models`. The service binds per `server_loop` attempt (a new `run_server_on` per retry).
- **Filesystem writes:**
  - `Config::save` (config.rs:395) is called by `resolve_config` on every CLI start (cli.rs:220), `for_service` (cli.rs:413), `add-model` (blue_onyx_prism.rs:287), `download-models --add-to-config` (:259) and `ConfigStore::update` (config_store.rs:288). All of these truncate and write in place.
  - The legacy rename is at config.rs:22.
  - Log directory creation and the daily log file are at cli.rs:499-501, with no retention.
- **Downloaded content parsed or extracted:** none in these files.
- **External programs:** `/usr/sbin/sysctl` (macOS, absolute path) and `powershell` (Windows, search path, no timeout) in system_info.rs.
- **Threads and channels:**
  - The tokio runtime has 2 workers plus the default blocking pool (512 max).
  - Per generation there is one "resources" watcher thread, which polls a crossbeam receiver every 250 ms and is joined in `retire`.
  - The service adds a "server" thread and an unbounded mpsc of `Event`; at most two messages are ever sent.
  - The test client runs a single-threaded runtime with `buffered(parallel)`.
- **Locks:** `ConfigStore` holds an RwLock across file IO (write, then read back). `StateHandle` uses a RwLock per model. The service's `Mutex<CancellationToken>` uses `lock_recover`.
- **Hot path:** none of these files is on the per-request detection path except the `StateHandle::get/is_ready` read lock (one clone of `ModelState` per call).

### From the non-Rust+tests (Opus subagent) report

- **Listeners**:
  - Tests bind `127.0.0.1:<free port>` (in-process `server::bind`, or the production `run_server_on`).
  - smoke.sh starts the release binary on ports 32168-32171 in sequence.
  - Docker publishes 32168 on all host interfaces. The systemd and launchd units use the config's `bind_address` (default `0.0.0.0`).
  - install_service.ps1 opens an inbound firewall rule for the port on every profile.
- **Filesystem write paths**:
  - `export_model.py` writes `--out-dir` (staging, in service mode) and `--work`, and chdirs into `--work`.
  - `build_bench_sets.py` writes `--work` and `assets/bench/*.json`.
  - `convert_onnx_to_ir.py` writes next to the ONNX file or into `--out`.
  - The install scripts write `/etc/systemd/system`, `/Library/LaunchDaemons` and `HKLM\...\ServicesPipeTimeout`.
  - Tests write `temp/bo_test_*` (never removed) and `temp/bop-ort-test-cache` (shared across runs).
- **Downloaded content parsed or executed**:
  - The Docker build runs `setup-onnxruntime` and `fetch --for-config` (hash-pinned).
  - CI and release run `setup-openvino`, `setup-onnxruntime` and `download-models` (hash-pinned).
  - `build_bench_sets.py` parses HF tree JSON, COCO JSON inside a pinned zip, and YOLO label txt (`int(p[0])` index into `EXDARK_NAMES`; a bad class id raises IndexError and exits non-zero).
- **External programs**:
  - `export_model.py` is run by the service as `python -u -I` in the pinned env (a user-triggered export only). It imports DEIMv2 source from the pinned `tool:deimv2-src` archive.
  - gen_locks.sh runs the pinned `uv`. smoke.sh runs the server, the client and `taskkill` (Windows).
- **Threads/channels**: not applicable to these files beyond the harness: one runner thread per `Server::start`, 30 s bounded shutdown waits.


## Not verified on this machine

### From the backend (Fable subagent) report

- Windows paths: DXGI enumeration, LUID matching, `SetEnvironmentVariableW` behavior, `LOAD_WITH_ALTERED_SEARCH_PATH` import resolution, `DirectML.dll` preload order vs System32, `PATH` prepending. Reasoned from source and the libloading 0.9.0 registry copy only.
- NVIDIA/CUDA paths: `cuda_driver::query` against a real driver, `preload_cuda_libs` ordering on a real `cuda-libs` set, whether `dlclose`/`FreeLibrary` of cudart/cuBLAS/cuDNN after a bare probe is harmless (BE-02), TensorRT engine cache behavior.
- Whether OpenVINO invokes the completion callback when `infer_async` fails synchronously (BE-14), and whether `ov_infer_request_free` waits for an in-flight async job in the 2026.x C wrapper (the Rust code additionally cancels+waits in `abort`, so the ordering is safe regardless).
- `openvino_sys::library::load_from` behavior on a second call or after a failed first `Core::new` (a later registry generation after `setup-openvino`): the generated loader was not inspected (`generated/functions.rs` and `mod.rs` showed no obvious `OnceLock`/`load_from` symbols in the grep). `preload_dependencies` itself only runs once per process (libs.rs:264), so a *different* OpenVINO install dir in a later generation reuses the first TBB/libopenvino.
- `loaded_openvino_c()`'s env-search fallback resolving a different `libopenvino_c` than openvino-sys loaded (libs.rs:337) — not reproducible here; both searches follow the same env vars.
- The `openvino` crate's `get_data` alignment assert (device.rs:876-888) for GPU-plugin output tensors on Intel hardware (no Intel GPU on this host).
- Behavior of the real worker on stale/unexpected completions (worker.rs is outside this review).

### From the pipeline (Fable subagent) report

- Windows reserved-name behaviour of `std::fs::write` for `CON.jpg` (PL-09) and DXGI/Windows-only paths: no Windows machine.
- Actual GPU/OpenVINO behaviour of `unstick` + `collect` on a hung device (PL-06): no OpenVINO GPU or NVIDIA here; reasoning from code only.
- `tests/pipeline.rs` was not run (ignored; needs `models/` and `openvino/`).
- Whether `analysisRoundTripMs` is filled by the server (outside my files).
- Memory numbers in PL-03/PL-04 are computed from sizes, not measured.

### From the resources (Fable subagent) report

- Windows: `File::try_lock` semantics across the service and CLI (same process family), `CREATE_NO_WINDOW` child spawning, `rename` over in-use DLLs (`in_use_hint` path), `std::fs::copy` link fallbacks in `extract.rs`, the Windows service stop path dropping `Process`/`Provisioner` (and therefore `Exporter::shutdown`) before exit; `std::process::exit(1)` in `src/bin/blue_onyx_prism.rs:334` was not traced.
- Whether reqwest 0.13.1's default policy follows `https`→`http` redirects was taken from its documentation/behaviour knowledge, not re-read from source on this machine (RS-01 is phrased as "no explicit guard").
- Real downloads/extractions of the pinned OpenVINO/ORT/CUDA archives, uv, and the DEIMv2 source (no network use in this review); `list-resources --check-urls` not run.
- The DEIMv2 `config.json` `eval_spatial_size` of pico/n/s/m/l/x equals 640 (the Rust side passes `--imgsz 640` for them; the script fails cleanly otherwise).
- That `GET /v1/resources` and the resource actions run on `server::blocking` and that the UI escapes `releaseNotes`/`log` lines (server/template files are out of scope).
- The `#[ignore]`d `tests/provision.rs::model_download_goes_initializing_to_ready_without_restart` (needs `models/` + OpenVINO under the test root).
- NVIDIA-specific paths (`system_cuda_present`, CUDA wheel extraction on Linux/Windows) – no NVIDIA hardware here.

### From the server+UI (Fable subagent) report

- Windows behaviour of `browse::roots()` drive probing, verbatim/UNC `display()` stripping, SYSTEM
  account exposure of other users' directories (no Windows machine here).
- hyper's header-read timeout being inert without a `Timer` under `axum::serve` 0.8.9 (deduced from the
  vendored sources, not exercised with a slowloris client).
- Memory growth of SV-04/SV-08 was reasoned from limits in the sources (hyper `DEFAULT_MAX_BUFFER_SIZE`,
  `DEFAULT_CAPACITY`), not measured.
- `tokio::spawn` from inside a `blocking` closure (`schedule_restart`) relies on the blocking thread
  having the runtime context; the existing tests exercise it (`config_models_selection`,
  `loglevel_and_restart` pass), so it is verified only on the test runtime.
- Chrome's Local Network Access / PNA preflight may block some cross-site POSTs to localhost from
  **HTTPS public** pages in recent Chrome versions; Firefox and Safari do not, and HTTP pages and DNS
  rebinding are unaffected, so SV-01 stands regardless.
- Behaviour of the worker's image saving with the unvalidated `save_image_path` (SV-03) was inferred from
  the config field semantics; the write code is in `worker.rs`/`image.rs` (another reviewer's files).

### From the benchmark-core (Opus subagent) report

- The Windows behavior of `std::fs::rename` over an open `benchmark.json` (antivirus or indexer handles), and drive-relative joins in `ImageSetInfo::image_path` for a `C:name` file entry.
- Real benchmark runs: no OpenVINO/ORT inference was exercised here (unit tests only). CUDA, TensorRT and DirectML device sweeps, and GPU compile time vs cancel latency, are unverified. So is whether GPU outputs can yield non-finite confidences that pass the threshold (another path to `null` in the file).
- Which conditions make `GpuAdapter.cuda` appear or disappear between runs on an NVIDIA Windows host (BM-03 trigger beyond version upgrade).
- Whether `actions::download` and `Job::into_dir(..).key()` can disagree (BM-06 trigger); the resources code is out of scope.
- The service shutdown and generation-change cancel path (`wait_idle` caller, runner cancel on generation stop) lives outside these files.

### From the benchmark-metrics (Opus subagent) report

- Windows behavior of `Path::join` with `C:x.jpg` (BQ-03) is from the std docs, not run here.
- Allocation-failure abort vs panic for BQ-01 not executed (would allocate terabytes); reasoning from `vec!` semantics.
- Size of `benchmark-preds.json` on a real multi-model, multi-device machine (BQ-04): estimated, not measured (local scratch copies are ~2 KB fixtures).
- Whether `benchmark.json`'s `image_sets` are replaced when a later run uses different datasets/`max_images`, which would silently drop stored runs' images from searches (search.rs:556-561) — depends on report.rs merge logic, out of scope.
- No NVIDIA/Windows: device-specific prediction differences (e.g. FP16 NaN) could not be exercised against the NaN paths of BQ-11.
- Built-in set downloads were not fetched; URLs/SHA-256 checked for presence/format only (the weekly `list-resources --check-urls` CI covers reachability).

### From the core/cli/config (Opus subagent) report

- **Windows service** (`blue_onyx_prism_service.rs`): not compiled here, because the cross-check fails in `aws-lc-sys` (no `windows.h`). SCM behavior, event-log layer, control codes 130 and 131, and PowerShell memory detection were all reviewed by reading only.
- **Signal handling on Linux under systemd** (SIGTERM, stop timeout): not run.
- **Runtime drop:** whether dropping the tokio runtime at the end of `main` blocks on outstanding `spawn_blocking` work (benchmark, export) during shutdown.
- **Coarse mtime filesystems** (CO-08): the behavior is inferred from the code and was not reproduced.
- **CO-03 crash:** confirmed from the crossbeam-channel 0.5.17 source (`flavors/array.rs:108-133`) and from std's `Duration` `Add` semantics. It was not reproduced, because that would mean editing a config.
- **CodeProject.AI label spellings** compared with `assets/coco_classes.yaml` (Darknet names).
- The `profile.rs` handling of `inflight`/`ort_sessions` bounds (`MAX_INFLIGHT` clamps) belongs to another reviewer. CO-03's statement that `inflight` is clamped there was not re-checked in depth.

### From the non-Rust+tests (Opus subagent) report

- Windows:
  - install_service.ps1 / uninstall_service.ps1 were not run. The behavior of the sc.exe 1072 race and of PowerShell 5.1 native exit codes is from documentation.
  - Whether a typical install dir is writable by non-admin users depends on where the archive is extracted.
  - The DirectML smoke on WARP and the Windows service binary were not run.
- Linux: install_systemd.sh, the systemd unit, the Docker image build/run, the `openvino` UID in `openvino/ubuntu24_runtime:2026.4.0` and `/opt/intel/openvino` in that image were not run or checked.
- No NVIDIA hardware, so CUDA/TensorRT and the `nvidia-cuda-libs` path were not run. The Pascal claim in NR-12 rests on the catalog comment (catalog.rs:92-97) and tests/select.rs, not on hardware.
- No Intel GPU, so tests/backend_cpu.rs's GPU branch was not run.
- I did not run the ignored tests or `cargo test`. The review was read-only by choice, to avoid writing `target/` contention with other reviewers.
- softprops/action-gh-release v3.0.3 update/overwrite semantics (NR-07) were not checked against its source.
- PyTorch CPU-index wheel availability for linux aarch64 at `torch==2.14.0+cpu`: gen_locks.sh `cmp`s the three resolutions, so they were identical when generated. Installability on aarch64 is not run here.
- macOS Gatekeeper behavior on the unsigned release binaries and downloaded dylibs (quarantine propagation through `tar`) was not checked.
- Provenance and license of `tests/data/dog_bike_car.jpg`, which is embedded in the binaries.


### Phase 3 notes (items from the brief without a finding ID)

- **Reload leak (found by the new test `reload_model_n_times_does_not_leak`):** RSS grew ~39 MB per
  model reload. Cause: `openvino::Node` (openvino crate 0.11) has no `Drop`, so the input/output
  ports taken in `backend::device::introspect` leaked and kept the whole read graph (weights
  included) alive. Fixed: a `Port` wrapper that frees the port (and its name string). Measured on
  macOS arm64, OpenVINO CPU, 10 reloads: before 345 -> 693 MB; after 314 -> 357 MB, flat over 40
  reloads. Threads stayed at 18 throughout (no thread leak).
- **Buffer reuse:** already present (`prep::TensorPool`, one `Preprocessor` per model geometry per
  prep thread; the image field is moved as `Bytes` without a copy). Not changed.
- **Config change log / metrics:** bounded already (50 change records; fixed histogram buckets).
  `benchmark.json` holds one result per model x device and no history. Not an issue.
- **Outbound HTTP timeouts:** the download client has connect (30 s) and read (60 s) timeouts, the
  release check 10 s. Not changed.
- **Disk:** saved images pruned by age (`save_image_max_age_days`, default 30) and folder size
  (`save_image_max_mb`, default 2048) by a per-process sweeper every 10 min; a warning when free
  space under the data root drops below 1 GiB (at most hourly); stale `.partial` files (> 7 days)
  and staging directories (> 1 h) removed at manager start; rolled logs capped at 14 (phase 2).
- **Shutdown:** the existing drain (5 s per generation, 1 s linger) is kept; the listener now
  stops through hyper's graceful shutdown with a 10 s deadline; the sweeper and watchdog threads
  are cancelled and joined. Not changed otherwise; the graceful path is exercised by
  `server::tests::slow_heads_and_idle_connections_are_closed` and the HTTP integration tests.
- **Startup / readiness:** `GET /v1/status` (open route) returns `ready`, `status`
  (ready / degraded / initializing / failed), per-model states and 200/503; the Docker
  `HEALTHCHECK` and compose healthcheck poll it (`--start-period=300s`); the systemd unit is
  `Type=notify` with `WatchdogSec=120` (`sdnotify.rs`: READY=1 once the listener is bound, the
  watchdog pings while the listener accepts connections, STOPPING=1 on shutdown).
- **Bounded compile wait:** not added. A compile is a blocking FFI call on the worker thread that
  cannot be aborted safely; a model stuck compiling shows as `initializing` on `/v1/status`, and
  the watchdog deliberately does not treat it as a hung process. Deferred to phase 4 as a
  "compiling for longer than N minutes" warning.
- **Not verified on this machine:** the systemd watchdog and `Type=notify` on Linux, the Docker
  healthcheck, Windows builds of `sdnotify`/`retention`, the listener timeouts under real Blue
  Iris load (unit-tested over loopback TCP only), the CoreML rewrite on a real > 512 MiB model.

## Per-finding status

Updated at the end of each phase (phase 1: branch `review/phase-1-crash-safety`; phase 2: branch `review/phase-2-security`; phase 3: branch `review/phase-3-resources-load`). Phase 0 leaves every finding open; the phase column is where the fix
is planned (1 crash safety, 2 security, 3 resources/load, 4 operability, 5 maintainability).

| ID | Severity | Phase | Status |
|---|---|---|---|
| BM-01 | crash-or-data-loss | 1 | fixed in phase 1: unloadable benchmark.json is moved aside, bad entries dropped with a warning |
| BQ-01 | crash-or-data-loss | 1 | fixed in phase 1: YOLO names map index bounded before allocation |
| BQ-02 | crash-or-data-loss | 1 | fixed in phase 1: unloadable benchmark-preds.json is moved aside, bad runs dropped |
| CO-01 | crash-or-data-loss | 1 | fixed in phase 1: --model matches resolved paths and appends instead of replacing the list |
| CO-02 | crash-or-data-loss | 1 | fixed in phase 1: Config::save writes .tmp, fsyncs and renames; unchanged config is not rewritten at start |
| CO-03 | crash-or-data-loss | 1 | fixed in phase 1: Config::validate ranges every numeric field; queue clamped at spawn; saturating timeout math |
| PL-07 | crash-or-data-loss | 1 | fixed in phase 1: worker_queue_size validated 0..=1024 and clamped at spawn |
| BE-01 | security-boundary | 2 | fixed in phase 2: Windows SetDefaultDllDirectories + AddDllDirectory for runtime dirs; CUDA probed by absolute path only (runtime behavior on Windows unverified here) |
| BE-04 | security-boundary | 2 | fixed in phase 2: no temp-dir fallback for the CoreML cache; a cache hit is byte-compared with the fresh rewrite |
| BM-08 | security-boundary | 2 | fixed in phase 2: Markdown report escapes HTML/Markdown in headings, paragraphs and cells |
| BQ-03 | security-boundary | 2 | fixed in phase 2: safe_file_name requires one Normal component (rejects `:` and prefixes) |
| CO-07 | security-boundary | 2 | fixed in phase 2: --allow-remote-admin/--force-cpu/--save-ref-image apply to the run only; remote admin logged at warn on start |
| NR-01 | security-boundary | 2 | fixed in phase 2: docker-config ships allow_remote_admin false; compose publishes on 127.0.0.1 by default; README explains the LAN case |
| NR-02 | security-boundary | 2 | fixed in phase 2: install_service.ps1 defaults to a virtual service account, resets the install dir ACL, checks exit codes, no password on the command line (not run here: no Windows) |
| NR-03 | security-boundary | 2 | fixed in phase 2: launchd runs as _blueonyx with data under /usr/local/var and newsyslog rotation (not run here) |
| NR-04 | security-boundary | 2 | fixed in phase 2: systemd unit hardened (NoNewPrivileges, ProtectSystem=strict, ProtectHome, PrivateTmp, ReadWritePaths, DevicePolicy, RestrictAddressFamilies, SystemCallFilter); binary root-owned, data in /var/lib |
| NR-05 | security-boundary | 2 | fixed in phase 2: contents: read by default in every workflow; contents: write only on the release job |
| PL-09 | security-boundary | 2 | fixed in phase 2: saved-image stems sanitized to [A-Za-z0-9._-], 64 chars, reserved device names fall back to a UUID |
| RS-01 | security-boundary | 2 | fixed in phase 2: https_only, custom redirect policy (https targets only, 5 hops), final URL checked |
| RS-07 | security-boundary | 2 | fixed in phase 2: env_clear plus an explicit allowlist for the export child |
| RS-16 | security-boundary | 2 | fixed in phase 2: directory listings only echoed under the data root; class YAML capped at 1 MiB; add-by-path limited to the browse roots |
| SV-01 | security-boundary | 2 | fixed in phase 2: same-origin check (Sec-Fetch-Site / Origin / Referer vs Host) on admin POST/PUT/PATCH/DELETE; header-less clients allowed (documented); router classification test |
| SV-03 | security-boundary | 2 | fixed in phase 2: config::validate_dir_setting on save_image_path/log_path/cache_dir (NUL, overlong, verbatim, UNC, drive-relative, .., system dirs); changes warn-logged |
| SV-05 | security-boundary | 2 | fixed in phase 2: /v1/config minimal for non-admin peers, no sync_config from remote |
| SV-09 | security-boundary | 2 | fixed in phase 2: nosniff, Referrer-Policy, X-Frame-Options, CSP (no inline scripts/styles; all page scripts moved to /static), Cache-Control no-store on JSON |
| SV-10 | security-boundary | 2 | fixed in phase 2: browse canonicalizes and requires a configured root (data root, models, downloads, home, Windows drives); symlink escapes refused |
| BE-02 | resource-exhaustion | 3 | fixed in phase 2: system_cuda_present cached per process (OnceLock), libraries stay loaded |
| BE-06 | resource-exhaustion | 3 | fixed in phase 3: CoreML rewrite skipped above 512 MiB (the original file is used); tensors and attributes parsed lazily; cached copy compared in chunks |
| BM-04 | resource-exhaustion | 3 | fixed in phase 1: --repeat 1..=1000, --warmup 0..=100 (same bounds as the service) |
| BM-11 | resource-exhaustion | 3 | fixed in phase 3: max_images capped at 2000 per run (spread over the sets); each model merged and saved as it finishes, only summaries kept; sweep clones summaries only |
| BQ-04 | resource-exhaustion | 3 | fixed in phase 3: stored predictions capped at 100 per image; parsed file cached by path+mtime+len; search handlers run on the blocking pool |
| BQ-05 | resource-exhaustion | 3 | fixed in phase 1: benchmark image files checked against a size limit before the read |
| CO-06 | resource-exhaustion | 3 | fixed in phase 2: log_path keeps the newest 14 daily files |
| NR-06 | resource-exhaustion | 3 | fixed in phase 2: launchd stdout log rotated by newsyslog |
| PL-01 | resource-exhaustion | 3 | fixed in phase 3: effective threshold floored at 0.01; at most 3000 NMS candidates (O(n) top-k) and 300 detections; NMS compares against kept boxes only |
| PL-03 | resource-exhaustion | 3 | fixed in phase 1: header-first decode with the max_image_pixels ceiling on both decoders; zero dims rejected |
| PL-04 | resource-exhaustion | 3 | fixed in phase 1: model input side <= 8192 and tensor <= 256 MiB, else the model fails to load |
| RS-02 | resource-exhaustion | 3 | fixed in phase 2: export children run in a process group (Unix) / Job Object (Windows) and the tree is killed on cancel, timeout and stop |
| RS-03 | resource-exhaustion | 3 | fixed in phase 2: 4 GiB per entry / 8 GiB per archive decompressed ceilings enforced while streaming; hostile entries refuse the archive |
| RS-05 | resource-exhaustion | 3 | fixed in phase 2: Content-Length / Content-Range / running count checked against the pin; mismatch is permanent, not retried |
| RS-08 | resource-exhaustion | 3 | fixed in phase 2: child output lines capped at 4 KiB, 2000 lines buffered |
| RS-12 | resource-exhaustion | 3 | fixed in phase 2: release-check body capped at 1 MiB, typed parse, fields truncated |
| SV-04 | resource-exhaustion | 3 | fixed in phase 3: hyper served with a timer, 10 s header-read/idle timeout, 10 s graceful deadline; process-wide in-flight cap (sum of queue capacities + 16, default 64) answers 503 and counts requests_rejected_total{reason="overloaded"}; 30 s multipart read timeout; bodies discarded (not buffered) when the model is full, failed, initializing or unknown |
| SV-08 | resource-exhaustion | 3 | fixed in phase 2 (name shortened) and phase 3: log ring messages capped at 4 KiB, targets at 256 bytes, ring at 8 MiB |
| BE-03 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BE-05 | correctness | 1 | fixed in phase 1: checked arithmetic in the Split/Flatten emitters; overflow skips the node with a reason |
| BE-07 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BE-09 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BE-14 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BM-02 | correctness | 1 | fixed in phase 1: --min-confidence/--iou/--accuracy-weight require a finite 0..=1 value |
| BM-03 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BM-10 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BQ-06 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BQ-07 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BQ-08 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BQ-09 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| CO-04 | correctness | 1 | fixed in phase 1: validate() runs at load, form, CLI merge (no write-back on error) and disk sync (not adopted) |
| CO-08 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| CO-09 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| CO-10 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| CO-14 | correctness | 5 | fixed in phase 1: test client uses total_cmp, nearest-rank p95, absolute start times with checked arithmetic |
| NR-07 | correctness | 4 | fixed in phase 2: release workflow refuses a manual run on an existing tag or published release |
| NR-08 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| NR-09 | correctness | 4 | fixed in phase 2: README Docker section matches the compose volumes |
| NR-10 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| NR-11 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| PL-02 | correctness | 1 | fixed in phase 1: non-finite detections dropped before NMS and serialization, scores clamped, coords clamped |
| PL-05 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| PL-08 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| RS-09 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| RS-15 | correctness | 4 | fixed in phase 2: resumed downloads are hashed whole; a changed upstream file fails permanently (test added) |
| SV-13 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| SV-14 | correctness | 4 | open (phase 0: identified, not yet fixed) |
| BE-08 | operability | 4 | open (phase 0: identified, not yet fixed) |
| BE-10 | operability | 4 | open (phase 0: identified, not yet fixed) |
| BE-13 | operability | 4 | open (phase 0: identified, not yet fixed) |
| BE-15 | operability | 4 | open (phase 0: identified, not yet fixed) |
| BM-05 | operability | 4 | open (phase 0: identified, not yet fixed) |
| BM-06 | operability | 4 | open (phase 0: identified, not yet fixed) |
| BQ-10 | operability | 4 | open (phase 0: identified, not yet fixed) |
| CO-05 | operability | 4 | open (phase 0: identified, not yet fixed) |
| CO-11 | operability | 4 | open (phase 0: identified, not yet fixed) |
| CO-12 | operability | 4 | open (phase 0: identified, not yet fixed) |
| CO-13 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-12 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-13 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-14 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-15 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-16 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-17 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-18 | operability | 4 | fixed in phase 2: Dockerfile HEALTHCHECK on /v1/vision/custom/list |
| NR-19 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-20 | operability | 4 | open (phase 0: identified, not yet fixed) |
| NR-21 | operability | 4 | open (phase 0: identified, not yet fixed) |
| PL-06 | operability | 4 | open (phase 0: identified, not yet fixed) |
| PL-11 | operability | 4 | open (phase 0: identified, not yet fixed) |
| RS-04 | operability | 4 | open (phase 0: identified, not yet fixed) |
| RS-06 | operability | 4 | open (phase 0: identified, not yet fixed) |
| RS-10 | operability | 4 | open (phase 0: identified, not yet fixed) |
| RS-11 | operability | 4 | open (phase 0: identified, not yet fixed) |
| RS-14 | operability | 4 | open (phase 0: identified, not yet fixed) |
| SV-06 | operability | 3 | fixed in phase 3: every benchmark handler's file/JSON/CPU work runs in server::blocking; image files served from a cached index |
| SV-07 | operability | 3 | fixed in phase 3: the browse walk and drive probing run in server::blocking |
| SV-11 | operability | 4 | fixed in phase 2: live.js handles non-JSON/403, capped exponential backoff, waitRestart gives up after 2 min, timers paused while hidden |
| SV-12 | operability | 4 | open (phase 0: identified, not yet fixed) |
| SV-15 | operability | 1 | fixed in phase 1: catch_panic middleware answers 500 (JSON or HTML) and the connection stays usable |
| BE-11 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| BE-12 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| BM-07 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| BM-09 | maintainability | 5 | fixed in phase 1: percentile/Stats handle empty slices and NaN |
| BQ-11 | maintainability | 5 | fixed in phase 1: roc::auc groups equal scores with total_cmp, terminates on NaN |
| BQ-12 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| BQ-13 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| CO-15 | maintainability | 5 | fixed in phase 1: crate-level lint policy in lib.rs and every binary; deny(unsafe_code) outside backend |
| CO-16 | maintainability | 5 | fixed in phase 1: one check_threshold rule (finite, 0..=1) on every path |
| CO-17 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| NR-22 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| NR-23 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| NR-24 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| NR-25 | maintainability | 5 | fixed in phase 2: deny.toml ignores carry a reason and a review date |
| NR-26 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| PL-10 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| PL-12 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| RS-13 | maintainability | 5 | open (phase 0: identified, not yet fixed) |
| SV-16 | maintainability | 5 | fixed in phase 2: message.html passes values via data attributes |
