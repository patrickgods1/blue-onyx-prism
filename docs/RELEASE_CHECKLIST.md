# Release checklist

Releases are built by `.github/workflows/release.yml` (Windows, Linux, macOS archives plus a
`.sha256` each) and published by hand from the draft it creates.

## Before tagging

1. CI is green on `main` (build-test on all three OSes, including the OpenVINO CPU,
   ONNX Runtime CPU, macOS CoreML smoke tests; the Windows DirectML smoke test is informational
   until it is made required).
2. Update `version` in `Cargo.toml` (and `Cargo.lock`) and merge to `main`. The tag must equal that
   version, or the release workflow fails.
3. Manual NVIDIA check (CUDA/TensorRT cannot run on hosted runners). The pinned Windows CUDA build
   of ONNX Runtime has kernels for compute capability 7.5+ only (`catalog.rs`), so on Windows use a
   Turing or newer card (GTX 16xx, RTX 20xx or later); the GTX 1070 Ti (Pascal, 6.1) works only
   under Linux, whose CUDA build includes sm_60. With the release build and
   `setup-onnxruntime --flavor cuda` installed (plus CUDA 12 / cuDNN 9, or
   `fetch --resource nvidia-cuda-libs`):

   ```sh
   # Linux
   ./blue-onyx-prism list-devices | grep -F 'ort:cuda'            # must be listed as runnable
   ./blue-onyx-prism benchmark --all-devices --report cuda.html
   grep -qF 'ort:cuda' cuda.html || { echo "FAIL: no ort:cuda row, CUDA was not exercised"; exit 1; }
   ```

   ```powershell
   # Windows (Turing or newer)
   .\blue-onyx-prism.exe list-devices
   .\blue-onyx-prism.exe benchmark --all-devices --report cuda.html
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
   .\blue-onyx-prism.exe benchmark --all-devices --report igpu.html
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
