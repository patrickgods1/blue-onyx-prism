# Changelog

All notable changes to Blue Onyx Prism are listed here, newest first. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/). Finding IDs (`RR-*`) point to
`docs/PRODUCTION_REVIEW.md`, which has the evidence behind each entry.

## [Unreleased]

The release-readiness review (phases 1-10, PRs #121-#131 and this phase), grouped by what you will notice.

### Upgrading

- Upgrading from 0.2.0 keeps every file: config, label files, sidecars, the review queue, training
  runs and tasks, benchmark results (checked by frozen 0.2.0 fixtures and an install-then-upgrade
  test). Training history moves out of `runs.json` into each run's `history.jsonl` at first start.
- Going back to 0.2.0 is not supported: 0.2.0 reads `runs.json` and `tasks.json` all or nothing, so
  one record of a kind it does not know (an overlaps task, say) makes it see an empty list, and its
  next save writes over the file. Copy the training folder (`<data root>/training`) first if you
  may go back.

### Added

- `check_updates` in the config: `false` stops the update check against GitHub.
- The Config page shows the licence of each catalog model's weights (IPcam-general: AGPL-3.0).
- `blue-onyx-prism diagnostics` and `GET /v1/diagnostics`: a zip to attach to a bug report.
- `test-blue-onyx-prism --rate --duration`: an open-loop load test.

### Security
- Admin pages and `GET /v1/config` refuse a `Host` that is not an IP address, `localhost` or an `allowed_hosts` name (DNS rebinding, RR-S1). An admin page opened by a name such as `http://nas.local:32168` answers 403 until the name is in `allowed_hosts`.
- Remote administration needs a token (`allow_remote_admin`): 256 bits in `<data root>/auth/admin-token` (moved there from `<data root>/admin-token`), shown by `blue-onyx-prism admin-token [--rotate]`, sent as `Authorization: Bearer` or through `/login` (RR-S4). Five wrong tokens a minute per address get 429. Detection routes never need it.
- A request that carries `X-Forwarded-For`, `Forwarded` or `X-Real-IP` is never treated as this machine, so a local reverse proxy cannot skip the token.
- `download_dir`, `onnxruntime_dir` and `openvino_dir` can no longer be changed from the web UI (they choose libraries the service loads). Moving the data folder means editing the config file and restarting.
- One-off flags (`--allow-remote-admin`, `--force-cpu`, `--save-ref-image`) are no longer written to the config file.
- Uploads: 10 s read deadline, a per-address limit on bodies still arriving and a 1 GiB byte budget across in-flight uploads (503 `upload_budget`). Blue Iris's total request count is never capped.
- URL import refuses internal addresses and hosts, by name and after resolving.
- Windows installer: protected ACL on the data folder, the service keeps only `SeChangeNotifyPrivilege`, and the firewall rule covers the local subnet on Domain and Private networks. After an upgrade, a PC on a Public network or Blue Iris on another subnet needs the manual rule in `packaging/windows/README.md`.
- Docker base images are pinned by digest; the Hugging Face token folder is protected on Windows.

### Data safety
- Uninstalling keeps your dataset, reviewed labels and training runs. Removing them takes `REMOVEDATA=1`, `--purge` or `apt purge`; `KEEPDATA=1` wins. An installed 0.2.0 still uninstalls by its old rules: upgrade first, or pass `KEEPDATA=1`.
- Auto-label no longer reports Done after failed writes and no longer deletes the only copy of its proposals.
- A damaged `runs.json` or `tasks.json` moves to `.corrupt-<time>` instead of being overwritten; one bad byte no longer empties an index.
- A lost Cancel is fixed (a job read as Queued just as it started kept running).
- Exports are built hidden and appear only when complete, and report lost frames and unreadable folders. A label read error is an error, not "no objects".
- `<dataset root>/.labels.lock` covers label writes from the CLI and the service together; a held lock is never given up on.
- An unavailable review folder is its own state on Review and the Train strip.

### Reliability
- Detection no longer stops silently after a GPU reset: submit errors take the health path and the supervisor reloads the model (RR-R1).
- A model on a fallback device retries its better device (5 min to 1 h) and swaps only on success.
- A reload or restart keeps the old worker serving until the replacement is Ready; if the new load runs out of memory the old one retires and the load is tried again.
- An overload 503 drains the upload so the answer arrives; abandoned warm-up backends are freed by bounded reaper threads.
- A runtime that lost a listed file is no longer shown as installed and downloads again.
- The benchmark runs each model's own device by default; several GPUs need `confirm_gpus=1` (CLI: `--confirm-gpus`), a queue job refuses it, and a breadcrumb names the last device after a hang. An empty `benchmark.devices` no longer sweeps every device.
- A bug in the label lock (a lock held over 5 s was treated as broken) is fixed.
- The test suite no longer has the flaky races found in phase 4; CI installs Python and node explicitly so a skipped test fails instead of passing.

### Speed
- `GET /v1/review/next` answers from a background index on folders of 2,000 frames or more: 1.3-3.2 ms at 200,000 frames instead of 12 s by walking (measured on an M1; Windows and spinning disks not measured).
- Review walks canonicalize once per day folder and read on up to 4 threads.
- Mounted drives (`/Volumes`, `/media`, `/mnt`) are browsable folders.

### Storage
- A daily sweep removes rebuildable files only: cache folders unused for 60 days, CoreML rewrites and compiled models of gone models, finished train staging. Labels, datasets, exports, models and runs are never touched.
- `review-queue.jsonl` is compacted under a cross-process lock.
- Training history moves out of `runs.json` into `<run>/history.jsonl` (an old file migrates at start); progress is saved at most every 5 minutes, so a 300-epoch run writes about 1 MB instead of hundreds.
- Config > Resources has a Storage card: what is safe to clear and what is your data.
- The service logs at Info with a 14-day, 500 MB cap.

### GPU memory
- Free GPU memory is read live (NVML, DXGI, amdgpu sysfs; shown in `GET /v1/devices` `gpuMemory`, Prometheus and the model load line). Training, auto-label and import warn before they would starve a loaded model; a labeler that cannot fit the card at all is refused.
- ONNX Runtime CUDA: `kSameAsRequested` arena, optional `ort_gpu_mem_limit_mb`, one extra session only for the default model. SigLIP runs on the CPU in auto-label when other models use the GPU.

### Polling
- Pages poll through one helper: chained, nothing while the tab is hidden, backoff on errors. `/v1/training` with nothing new is 2.9 KB instead of 62 KB (and 19 KB instead of 1.2 MB with many runs); the benchmark page's run poll is about 0.7 KB instead of 441 KB.
- Config readers never wait for a save's fsync.

### Structure
- The COCO tables and name normalisation moved to `model/coco.rs` and `model/name.rs`, ending a module cycle; the review routes were split into smaller files. No behaviour change.
- The `overlap_live_boxes` field of `GET /v1/review/labels` fixes a relabelled twin being dropped from overlap suggestions (RR-T9); `overlap_live` stays for an old page.
- The verify lane reads a typed drop kind, not the message text; the Blue Iris answer is byte-identical.

### Documentation and release
- Privacy section in the README listing every outbound connection and how to turn each off; the default model's AGPL-3.0 license is stated plainly in the README and shown on the Config page's Resources list.
- `THIRD_PARTY_LICENSES.md` crate table is generated from `Cargo.lock` by `scripts/gen_third_party.py`; CI fails when it differs.
- `SECURITY.md`, issue templates and this changelog. Windows: 11 is tested, 10 22H2 should work and is untested.

## [0.2.0] - 2026-10-27

Previous release; its notes are on the
[GitHub release](https://github.com/patrickgods1/blue-onyx-prism/releases/tag/0.2.0).

## [0.1.0] - 2026-10-15

First tagged release.

[Unreleased]: https://github.com/patrickgods1/blue-onyx-prism/compare/0.2.0...HEAD
[0.2.0]: https://github.com/patrickgods1/blue-onyx-prism/compare/0.1.0...0.2.0
[0.1.0]: https://github.com/patrickgods1/blue-onyx-prism/releases/tag/0.1.0
