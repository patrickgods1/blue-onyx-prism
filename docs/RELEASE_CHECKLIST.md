# Release checklist

Releases are built by `.github/workflows/release.yml` (Windows, Linux, macOS archives plus a
`.sha256` each) and published by hand from the draft it creates.

## Before tagging

1. CI is green on `main` (build-test on all three OSes, including the OpenVINO CPU,
   ONNX Runtime CPU, macOS CoreML smoke tests; the Windows DirectML smoke test is informational
   until it is made required).
2. Update `version` in `Cargo.toml` (and `Cargo.lock`) and merge to `main`. The tag must equal that
   version, or the release workflow fails.
3. Manual NVIDIA check (CUDA/TensorRT cannot run on hosted runners). On the GTX 1070 Ti box, with the
   release build and `setup-onnxruntime --flavor cuda` installed:

   ```powershell
   blue-onyx-prism benchmark --all-devices --report cuda.html
   ```

   Open `cuda.html` and confirm that every device's agreement with the CPU reference is `ok`.
   Keep the report for the release (step 6).

   Also check readiness: `curl -fsS http://127.0.0.1:32168/v1/status` returns 200 with
   `"ready": true` once the models are loaded (503 with `"status": "initializing"` while they
   compile), and the Docker image reports `healthy`.

## Tagging

4. `git tag X.Y.Z && git push origin X.Y.Z` (plain `X.Y.Z`, no `v` prefix). The workflow builds the
   archives and creates a **draft** release named after the version. `workflow_dispatch` does the
   same for the `Cargo.toml` version without a tag, and refuses to run once the tag `X.Y.Z` exists
   (it would build `main` HEAD, not the tagged commit). Any run refuses when release `X.Y.Z` is
   already published, so published archives and their `.sha256` files are never replaced; to redo
   a published release, bump the version. Only the final `release` job gets a write token; the
   build jobs run read-only.
5. Check the draft: three archives and their `.sha256` files are attached, and the generated release
   notes are sensible.

## Publishing

6. Attach `cuda.html` to the draft release, edit the notes, and publish it.
