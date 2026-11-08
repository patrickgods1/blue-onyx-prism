# Release checklist

Releases are built by `.github/workflows/release.yml` (Windows, Linux, macOS archives plus the
native installers: MSI, deb, rpm and pkg, with a `.sha256` each) and published by hand from the
draft it creates. Each build job installs, checks and uninstalls its installers on the runner
(`packaging/smoke-test-installers.sh`); a failed smoke test fails the release.

## Keeping dependencies current

The service never fetches "latest": every download is a pin (URL, size, SHA-256) in
`src/resources/catalog/`, and the Python environments install hashed locks. Updates reach users as
reviewed pin bumps in a release; an existing install sees the new version as an update (a loaded
runtime is swapped in at the next start). Two weekly jobs (`.github/workflows/check-urls.yml`) say
when a bump is due, each in one issue kept current:

- **Upstream updates available** (`scripts/ci/check_deps.py upstream`): catalog runtimes, uv, VC++
  runtime and the Python pins with a newer release. Pins held back on purpose (`HOLDS`: ONNX Runtime
  1.24.x for DirectML, cuDNN < 9.11 for Pascal) are listed apart.
- **Dependency advisories**: `cargo deny check advisories` on `main` and OSV advisories against every
  Python lock (`check_deps.py audit`).

Bumping a pin:

1. Catalog download: take URL, size and SHA-256 from the upstream file itself (download it, hash
   it), never from a web page alone; update every platform's part together. Run
   `cargo run -- list-resources --check-urls` and `cargo test resources`. OpenVINO: bump the
   Dockerfile's `openvino/ubuntu24_runtime` tag and digest in the same commit (CI checks the tag).
2. Python: change the `==` pin in `scripts/requirements-model-export.txt` / `-train.txt`, regenerate
   the locks with the pinned uv (`scripts/model-export/gen_locks.sh`, `gen_train_locks.sh`, a cutoff at
   least two weeks old), and for ultralytics / transformers the catalog constants too
   (`check_deps.py consistency` in CI checks all of these). Run a model export and a short training on
   one platform before merging.
3. Rust toolchain: `rust-toolchain.toml`, `Cargo.toml` `rust-version` and every workflow
   `toolchain:` line together (CI checks they agree).

## Before tagging

1. CI is green on `main` (build-test on all three OSes, including the OpenVINO CPU,
   ONNX Runtime CPU, macOS CoreML smoke tests; the Windows DirectML smoke test is informational
   until it is made required). `release.yml` checks this itself: its first job (`ci-green`,
   `packaging/release/wait-ci-green.sh`) refuses to build unless every check of the tagged commit
   passed, waiting up to 80 minutes for running ones. Tag only a commit whose CI already finished
   green, or the release waits, then refuses.
2. Update `version` in `Cargo.toml` (and `Cargo.lock`) and merge to `main`. The tag must equal that
   version, or the release workflow fails.
3. Manual NVIDIA check (CUDA/TensorRT cannot run on hosted runners). The pinned CUDA build of ONNX
   Runtime (1.31.0, `catalog::ORT_CUDA_BUILDS`) has kernels for compute capability 6.1, 7.5, 8.6, 8.9 and
   12.0 on Windows (6.0-12.0 without 10.0 on Linux), so the GTX 1070 Ti (Pascal, 6.1) is the Windows
   check; TensorRT needs Turing or newer. When bumping the pin, re-check the shipped provider library's
   fatbinaries (see the catalog comments) and that the cuDNN pin stays below 9.11 while Pascal is listed.
   With the release build and `setup-onnxruntime --flavor cuda` installed (plus CUDA 12 / cuDNN 9, or
   `fetch --resource nvidia-cuda-libs`):

   ```sh
   # Linux
   ./blue-onyx-prism list-devices | grep -F 'ort:cuda'            # must be listed as runnable
   ./blue-onyx-prism benchmark --all-devices --confirm-gpus --report cuda.html   # refused without the flag on two GPU devices
   grep -qF 'ort:cuda' cuda.html || { echo "FAIL: no ort:cuda row, CUDA was not exercised"; exit 1; }
   ```

   ```powershell
   # Windows (Turing or newer)
   .\blue-onyx-prism.exe list-devices
   .\blue-onyx-prism.exe benchmark --all-devices --confirm-gpus --report cuda.html   # refused without the flag on two GPU devices
   if (-not (Select-String -Path cuda.html -SimpleMatch 'ort:cuda' -Quiet)) { throw "FAIL: no ort:cuda row, CUDA was not exercised" }
   ```

   The step fails, and the release waits, when the report has no `ort:cuda` row with results:
   DirectML or OpenVINO rows alone do not count. Open `cuda.html` and confirm that every device's
   agreement with the CPU reference is `ok` (`ort:tensorrt` too when its libraries are
   installed). Keep the report for the release (step 8).
4. Manual Intel iGPU check (the primary target; hosted runners have no Intel GPU, so CI only
   covers the CPU fallback). On the Windows 11 i5-8500 (UHD 630) box, with the release archive
   (it bundles OpenVINO):

   ```powershell
   .\blue-onyx-prism.exe list-devices                     # openvino:gpu runnable, and the auto pick
   .\blue-onyx-prism.exe benchmark --all-devices --confirm-gpus --report igpu.html   # openvino:gpu and ort:directml count as two
   if (-not (Select-String -Path igpu.html -SimpleMatch 'openvino:gpu' -Quiet)) { throw "FAIL: no openvino:gpu row" }
   ```

   Confirm the `openvino:gpu` rows agree with the CPU reference (`ok`) and are faster than
   `openvino:cpu`, and that a second start loads from `cache/` in seconds. Keep `igpu.html` for
   step 8.

   Also check readiness: `curl -fsS http://127.0.0.1:32168/v1/status` returns 200 with
   `"ready": true` once the models are loaded (503 with `"status": "initializing"` while they
   compile), and the Docker image reports `healthy`.
5. YOLO26 (manual: its weights are AGPL-3.0, so CI never exports them and the YOLO26 parts of the
   integration tests are skipped there). Export `yolo26n` and `yolo26s` to IR with the Python env
   (`scripts/export_model.py --name yolo26n yolo26s --out-dir <root>/models`) and run the real-model
   tests against them: `BLUE_ONYX_TEST_ROOT=<root> cargo test --release -- --ignored --test-threads=1`
   (`<root>` also holds `openvino/` and `onnxruntime/`). `families_http` and `multi_model_http`
   must report the YOLO26 parts as run, not "not present: skipping".

## When a CI job fails with no failed step

The hosted macOS runners sometimes have no capacity. It looks like this:

- `build-test (macos-latest)` shows **no failed step, in fact no steps at all**: `gh run view <id>
  --json jobs` gives it `"conclusion": "cancelled"` and an empty `steps` list, while the run as a
  whole is `failure`.
- The job's annotation says `The job was not acquired by Runner of type hosted even after
  multiple attempts`, usually with `Due to capacity constraints, jobs targeting macOS arm64
  runners may experience longer queue times.` (seen in runs 41888292694, 41893062111, 41898876784).

That is GitHub's, not this repository's: `gh run rerun <id> --failed` is the right answer **for this
case only**. A job with a failed step is a real failure until shown otherwise. Do not rerun it to
see whether it goes away: read the step's log, reproduce it, and fix the cause.

`The hosted runner lost communication with the server` is a different case: the annotation says a
workflow can cause it by killing or starving the runner, so it is not GitHub's until shown otherwise.
It leaves no step log, so look at what the step could do to the machine first (RR-T11 in
`docs/PRODUCTION_REVIEW.md`).

A step that reports it timed out hit its `timeout-minutes` in `ci.yml`, at least twice that
step's slowest green run. Treat it as a hang: the `has been running for over 60 seconds` lines of
the log name the test.

## Tagging

6. `git tag X.Y.Z && git push origin X.Y.Z` (plain `X.Y.Z`, no `v` prefix). The workflow builds the
   archives and creates a **draft** release named after the version. `workflow_dispatch` does the
   same for the `Cargo.toml` version without a tag, and refuses to run once the tag `X.Y.Z` exists
   (it would build `main` HEAD, not the tagged commit). Any run refuses when release `X.Y.Z` is
   already published, so published archives and their `.sha256` files are never replaced; to redo
   a published release, bump the version. Only the `attest` job (OIDC token, attestation write)
   and the final `release` job (contents write) get more than a read token; the build jobs run
   read-only. The build stamps the binaries with the commit (`BLUE_ONYX_PRISM_GIT_SHA`, 12 hex
   digits) and the UTC build date (`BLUE_ONYX_PRISM_BUILD_DATE`): `blue-onyx-prism --version` and
   the `build_info` metric show them.
7. Check the draft: three archives and their `.sha256` files are attached, each archive contains
   `LICENSE` and `THIRD_PARTY_LICENSES.md` (plus the runtimes' own license files under
   `openvino/` and `onnxruntime/`), the `attest` job succeeded, and the generated release notes
   are sensible.

## Publishing

8. Attach `cuda.html` and `igpu.html` to the draft release, edit the notes, and publish it.

## Verifying a download (for users; link it from the release notes)

Each archive has a `.sha256` file and a signed build-provenance attestation that ties it to this
repository's release workflow and commit:

```sh
sha256sum -c blue-onyx-prism-X.Y.Z-linux-x86_64.tar.gz.sha256        # macOS: shasum -a 256 -c ...
gh attestation verify blue-onyx-prism-X.Y.Z-linux-x86_64.tar.gz --repo patrickgods1/blue-onyx-prism
```

```powershell
# Windows: compare the hash with the .sha256 file, then check the attestation (GitHub CLI)
(Get-FileHash blue-onyx-prism-X.Y.Z-windows-x86_64.zip -Algorithm SHA256).Hash.ToLower()
Get-Content blue-onyx-prism-X.Y.Z-windows-x86_64.zip.sha256
gh attestation verify blue-onyx-prism-X.Y.Z-windows-x86_64.zip --repo patrickgods1/blue-onyx-prism
```

`sha256sum -c` prints `OK`; `gh attestation verify` prints `Verification succeeded!`. Either
failing means the file is not the one the workflow built: do not run it.
