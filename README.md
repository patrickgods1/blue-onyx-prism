# Blue Onyx Prism

Blue Iris / CodeProject.AI compatible object detection service, written in Rust on native
[OpenVINO](https://github.com/openvinotoolkit/openvino) and [ONNX Runtime](https://onnxruntime.ai/).
Runs on Intel integrated and discrete GPUs (OpenVINO), NVIDIA GPUs (CUDA/TensorRT), any DirectX 12 GPU on
Windows (DirectML), Apple silicon (CoreML) and on CPU everywhere (Windows x86_64, Linux x86_64, macOS arm64).
The default device `auto` picks the best option for the detected hardware; see
[Hardware and device selection](#hardware-and-device-selection) and [docs/PLAN.md](docs/PLAN.md).

Formerly **Blue Onyx OpenVINO**. Existing `blue_onyx_openvino_config*.json` files are renamed to
the new names on first start, and `scripts/install_service.ps1` removes the old
`BlueOnyxOpenVINOService`. Prometheus metrics are now prefixed `blue_onyx_prism_`.

Modeled on [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT), adding multi-model serving
(`/v1/vision/custom/{model}`), YOLO26 end-to-end models and Intel GPU inference on Windows.

## Features

- Drop-in CodeProject.AI / DeepStack compatible API for Blue Iris (`/v1/vision/detection`, `/v1/vision/custom/{model}`, `/v1/vision/custom/list`).
- Native OpenVINO inference: Intel integrated/discrete GPU on Windows and Linux, CPU everywhere, automatic GPU -> CPU fallback.
- ONNX Runtime as a second runtime (NVIDIA CUDA/TensorRT, DirectML, CoreML, CPU) with `auto` device selection by detected hardware.
- Several models served at once, each with its own worker and metrics; per-model device, threshold, class filter and lazy loading.
- Model families: YOLO26 / YOLOv10 (end-to-end), YOLOv5 (the blue-onyx IPcam models), YOLOv8/9/11/12, RT-DETRv2 and DEIMv2, D-FINE, RF-DETR. ONNX or OpenVINO IR.
- Compiled-model cache (fast restarts), HTTP served while models compile, web UI (stats, test page, config editor), Prometheus metrics.
- In-app training of custom models on your own reviewed frames (YOLO26 / YOLO11 / YOLOv8, RF-DETR, RT-DETRv2, D-FINE) on an NVIDIA GPU or Apple silicon, with a held-out benchmark ([docs/TRAINING.md](docs/TRAINING.md)).
- Footage import (videos, Blue Iris clip and alert folders, a video URL with a licence confirmation, pinned MEVA clips) with scene-change, dHash and SigLIP 2 k-center frame selection, and auto-labelling for review (MM-Grounding-DINO + RF-DETR L fused, checked by Qwen3-VL and SigLIP 2; self-training re-label with a trained model).
- Runs as a Windows service, systemd unit, launchd daemon or Docker container.

## Supported platforms

| Platform | Inference | Notes |
|---|---|---|
| Windows 11 x86_64 | Intel GPU (iGPU/Arc) + CPU | Primary target; Windows service. Needs the MSVC runtime redistributable. |
| Linux x86_64 | Intel GPU (`/dev/dri`) + CPU | Install `intel-opencl-icd`; user in the `render` group. systemd and Docker files provided. |
| macOS arm64 | CPU, CoreML | launchd daemon. No OpenVINO GPU plugin exists for Apple silicon; CoreML runs through ONNX Runtime. |

NVIDIA (CUDA/TensorRT) and DirectML devices use ONNX Runtime; see the matrix below.

## Hardware and device selection

`"device": "auto"` (the default) chooses per model, tries the next candidate if compile or warm-up
fails, and always ends on the CPU. Ranking: discrete GPUs first (larger VRAM first), each with its
runtimes in order: NVIDIA `ort:cuda`, `ort:tensorrt`, then `ort:directml` (Windows); Intel Arc
`openvino:gpu.N`, then `ort:directml`; AMD/other `ort:directml` (Windows). Then integrated GPUs (Intel
via OpenVINO; DirectML on Windows for the others), Apple silicon (CoreML), and the CPU (`openvino:cpu`,
else `ort:cpu`). An NVIDIA GPU the ONNX Runtime CUDA build has no kernels for skips CUDA: the 1.31.0 CUDA 12
build covers compute capability 6.1 (GTX 10-series), 7.5, 8.6, 8.9 and 12.0 on Windows, and 6.0 through
12.0 (10.0 excepted) on Linux; `list-devices` and the device dropdown show the reason, and an installed
older build that lacks the kernels is offered as an update. TensorRT 10 needs 7.5+ (Turing or newer), and
a driver that cannot run CUDA 12 (older than R525) skips both. Such a card starts at DirectML on Windows.

When `benchmark.json` (written by `benchmark` next to the config) has results for a model, `auto` orders
that model's devices by measured speed instead: the devices `benchmark` would recommend (ran, agree with
the CPU reference, as accurate as the best) fastest first, then the remaining options in the ranking
above, CPU last. Results for a device that is gone or cannot run now are ignored; the file is read when
the models are (re)loaded. `list-devices` marks such a model's order "(benchmark order)".

| OS | GPU | `auto` picks | Setup | Notes |
|---|---|---|---|---|
| Windows | Intel iGPU/Arc | `openvino:gpu` | `setup-openvino` | Primary target. `ort:directml` is selectable. |
| Windows | NVIDIA | `ort:cuda` | `setup-onnxruntime --flavor cuda` | CUDA 12.8 and cuDNN 9.10 come from `nvidia-cuda-libs` (1.9 GB, `allow_large_downloads`) or your own install. Without them, or with the DirectML flavor loaded, `auto` is `ort:directml`, ahead of an Intel iGPU. GTX 10-series (Pascal, 6.1) cards run CUDA since the 1.31.0 build; TensorRT needs Turing or newer. |
| Windows | AMD / other DX12 | `ort:directml` | `setup-onnxruntime --flavor directml` | |
| Linux | Intel | `openvino:gpu` | `setup-openvino` | `intel-opencl-icd`, `render` group. |
| Linux | NVIDIA | `ort:cuda` | `setup-onnxruntime --flavor cuda` | CUDA 12.8 + cuDNN 9.10 from `nvidia-cuda-libs` or your own install; Pascal (6.0) and newer. |
| Linux | AMD | `openvino:cpu` | `setup-openvino` | No ROCm/MIGraphX in the stock ONNX Runtime packages: CPU. |
| macOS arm64 | Apple GPU | `ort:coreml` | `setup-onnxruntime` | RT-DETRv2 runs on CoreML (~1.6-1.9x `openvino:cpu` on an M1) because its batch dimension is pinned to 1; an RT-DETR export with an unnamed dynamic dimension is refused there (MPSGraph would abort) and uses the CPU. YOLO26 needs the `.onnx` export. |
| any | none | `openvino:cpu` | `setup-openvino` | `ort:cpu` if OpenVINO is missing. |

TensorRT (`ort:tensorrt`, first engine build takes minutes) follows CUDA in `auto` only when its
libraries load, so it is tried only if CUDA fails. The NPU (`openvino:npu`) can be selected but is never
chosen by `auto`. ONNX Runtime needs `.onnx` models; an IR `.xml` model uses a
sibling `<stem>.onnx` if present, otherwise its ONNX Runtime options are skipped.

Device specs: `auto`, `openvino:gpu[.N]`, `openvino:cpu`, `openvino:npu`, `ort:cuda[:N]`,
`ort:tensorrt[:N]`, `ort:directml[:N]`, `ort:coreml`, `ort:cpu`. The old `GPU`, `GPU.N` and `CPU` still
work and mean `openvino:*`. Set it globally (`device`), per model (`models[].device`) or with `--device <spec>`.

```sh
blue-onyx-prism list-devices         # detected GPUs, runnable options with reasons, and the auto pick
curl http://127.0.0.1:32168/v1/devices   # the same as JSON
```

The Config page has a device dropdown ("Auto - currently: ...", then the runnable options; options that
cannot run are greyed out with the reason).

Measured on an Apple M1 (`inferenceMs`):

| Model | `openvino:cpu` | `ort:cpu` | `ort:coreml` |
|---|---|---|---|
| IPcam-general | 37 | 69 | 17 (104 without the rewrite) |
| rt-detrv2-s | 101 | 235 | not run (excluded) |

CoreML graph rewrite: the CoreML execution provider of ONNX Runtime 1.24 has no HardSigmoid/HardSwish
and takes `Split` only from opset 13, so the YOLOv5 models (IPcam-*, delivery, package, ipcam-bird)
used to fall apart into ~50 CoreML pieces with CPU hops between them and ran slower than the CPU. On
`ort:coreml` the service therefore loads a rewritten copy with exact equivalents
(`HardSigmoid` -> `Mul`/`Add`/`Clip`, `HardSwish` -> the same times `x`, `Split` -> `Slice`s): one
CoreML partition, ~6x faster, outputs bit-identical on the CPU. The copy is written once to
`cache/coreml/<model>-<sha256 prefix>-v<N>.onnx` (the original file is never touched, and the model is
still reported under its own path); models with external data are left alone. Set the environment
variable `BOP_COREML_REWRITE=0` to load the original instead.

## First run

A fresh install can be just the binary and a config that names a model. On start the service works
out what the config and the hardware need, downloads only what is missing in the background, and
serves HTTP right away (models answer `Model initializing` with the download progress, e.g.
`downloading OpenVINO runtime 42% (18/44 MB)`, until they are ready):

```json
{ "device": "auto", "models": [ { "path": "models/IPcam-general.onnx" } ] }
```

- **What gets downloaded:** the model files (if the name matches the catalog, `list-models`), the
  runtime of the best device for this machine (`auto`: CUDA on NVIDIA, OpenVINO GPU on Intel,
  DirectML on other Windows GPUs, CoreML on Apple silicon, else CPU) and the runtime of its CPU
  fallback. On a Mac that is the model (29 MB), ONNX Runtime CoreML (31 MB) and OpenVINO (44 MB).
- **When:** models load on a CPU option as soon as one is installed, then the server restarts its
  registry (in the same process, a few seconds) to switch to the better runtime when that arrives.
  A second ONNX Runtime flavor (a process can load only one) is downloaded next to the first and
  used after the process restarts.
- **Where:** next to the executable (`openvino/`, `onnxruntime/<flavor>/` with `active.txt` naming
  the active flavor, `onnxruntime/cuda-libs/`, `models/`), or under `download_dir`. Every file is
  pinned by URL, size and SHA-256 in the binary (`src/resources/catalog/`), downloads resume after
  an interruption, and only whitelisted files are extracted. Downloaded libraries (OpenVINO, ONNX
  Runtime, DirectML, the NVIDIA CUDA libraries) are SHA-256 pinned and loaded into the service's
  process, so they run with its rights; no downloaded program is run except by a model export you
  start (see [Exporting models](#exporting-models)).

| Setting | Default | Meaning |
|---|---|---|
| `auto_download` | `true` | Download what the config needs at startup. `false` (air-gapped, Docker): only report it; affected models are `Failed` with the command to run |
| `allow_large_downloads` | `false` | Allow downloads over 500 MB, in practice the NVIDIA CUDA libraries |
| `download_dir` | `null` | Root for runtimes and relative model paths (default: the executable's directory) |

Commands (the same code the service uses):

```sh
blue-onyx-prism list-resources          # what is installed, needed by the config, or available, with sizes
blue-onyx-prism fetch --for-config      # download everything the config needs now (default without flags)
blue-onyx-prism fetch --resource model:ipcam-bird --resource onnxruntime-cpu
blue-onyx-prism fetch --all-for-platform  # everything for this OS/arch (large ones need --allow-large)
```

- **Offline / air-gapped:** set `"auto_download": false`, run `fetch --for-config` on a machine with
  network (same OS/arch and config), then copy the whole directory. `list-resources` on the target
  shows anything still missing.
- **Web UI:** the Config page has a *Resources* card, grouped into Runtimes, GPU libraries, Models
  by family (YOLOv5, RT-DETRv2, D-FINE, RF-DETR) and *Local models (not in config)* (`.onnx` / `.xml`
  files in `models_dir` that no config entry uses, e.g. exports made by hand), with state, size,
  Download / Remove, a live progress bar and *Add to config*. `GET /v1/resources` returns the same as
  JSON (each row has a `group`; local files are in `localModels`).
  Device options that need a download show "will download ... (size)" in the Device dropdown;
  choosing one and saving starts the download and switches when it is installed.
- **Versions:** the `cpu`, `directml` and `coreml` flavors are ONNX Runtime 1.24.4, the last release
  Microsoft publishes with the DirectML provider (the 1.25+ Windows packages have none; DirectML itself is
  in sustained engineering). The `cuda` flavor is 1.31.0 (the `gpu_cuda12` archive), whose Windows build
  has Pascal kernels again. A process loads one flavor, so the versions need not match; an installed
  flavor older than the catalog's shows *update to ...* on the Resources card and `list-resources`.
- **NVIDIA CUDA libraries (opt-in):** `ort:cuda` needs CUDA 12.8 and cuDNN 9. If they are not
  installed, the `nvidia-cuda-libs` resource (about 1.9 GB on Windows, 1.7 GB on Linux: cudart,
  cuBLAS, cuDNN 9.10 (the last with kernels for GTX 10-series cards), cuFFT, cuRAND, nvJitLink, NVRTC
  from NVIDIA's PyPI wheels) provides them. It is
  only downloaded with `allow_large_downloads`, `fetch --allow-large` or
  `fetch --resource nvidia-cuda-libs`, and the libraries are covered by the
  [NVIDIA Software License Agreement](https://docs.nvidia.com/cuda/eula/) (CUDA, cuDNN
  redistributables). TensorRT is never downloaded.
- **Exported models**: Ultralytics YOLOv8/v9/v10/11/12/26 (AGPL-3.0, cannot be redistributed),
  RF-DETR large (current 704 px release) and DEIMv2 are only published as PyTorch weights, so the
  service exports them to ONNX on request (Config page, *Models · exported on this machine*, or
  `fetch --resource model:yolo11s --allow-large`; see [Exporting models](#exporting-models)). A
  config entry pointing at such a file that is not exported yet (or is outdated, like the earlier
  `rfdetr-large` download) shows "needs export"; nothing is exported automatically.

## Install with an installer

Each [release](https://github.com/patrickgods1/blue-onyx-prism/releases) also ships native
installers that register the background service. They are small: the service downloads OpenVINO,
ONNX Runtime and the default model on its first start (`auto_download`), so the machine needs
internet access the first time, and the web UI shows the download progress.

| OS | File | Program | Config and data |
|---|---|---|---|
| Windows 10/11 x64 | `blue-onyx-prism-<version>-windows-x86_64.msi` | `C:\Program Files\Blue Onyx Prism` | `C:\ProgramData\BlueOnyxPrism` |
| Debian / Ubuntu 22.04+ | `blue-onyx-prism_<version>-1_amd64.deb` | `/opt/blue-onyx-prism` | `/var/lib/blue-onyx-prism` |
| Fedora / RHEL 9+ | `blue-onyx-prism-<version>-1.x86_64.rpm` | `/opt/blue-onyx-prism` | `/var/lib/blue-onyx-prism` |
| macOS (Apple silicon) | `blue-onyx-prism-<version>-macos-aarch64.pkg` | `/usr/local/blue-onyx-prism` | `/usr/local/var/blue-onyx-prism` |

After installing, open `http://127.0.0.1:32168/`. The installers are not code-signed yet:

- **Windows:** SmartScreen may show "Windows protected your PC". Click *More info*, then *Run anyway*.
  The MSI installs the `BlueOnyxPrismService` service (runs as Local Service, starts automatically),
  a firewall rule for TCP 32168 (`msiexec /i <file>.msi PORT=<n>` for another port's rule), and a
  Start Menu shortcut to the web UI. Silent install: `msiexec /i <file>.msi /qn`. Uninstall from
  *Settings > Apps*: it removes everything, including `C:\ProgramData\BlueOnyxPrism` (config,
  downloads, logs); to keep that folder for a reinstall, uninstall with
  `msiexec /x <file>.msi KEEPDATA=1` instead. Upgrading to a newer MSI always keeps it. If you installed the service earlier with `scripts\install_service.ps1`,
  run `scripts\uninstall_service.ps1` first; the MSI refuses to install over it. The console
  `blue-onyx-prism.exe` in Program Files still looks for a config next to itself, so pass
  `--config C:\ProgramData\BlueOnyxPrism\blue_onyx_prism_config_service.json` to work on the
  service's config (stop the service first).
- **Linux:** `sudo apt install ./blue-onyx-prism_*.deb` or `sudo dnf install ./blue-onyx-prism-*.rpm`.
  The package creates the `blueonyx` user and enables `blue-onyx-prism.service`. `apt remove` keeps
  the data and the user for a reinstall, `apt purge` removes everything; `dnf remove` removes
  everything (RPM has no purge).
- **macOS:** macOS blocks the unsigned package on first open. Open *System Settings > Privacy &
  Security*, click *Open Anyway* next to the message about the package, and run it again. To
  uninstall, open *Uninstall Blue Onyx Prism* in *Applications*: choose *Keep My Data* (config,
  models and logs stay for a reinstall) or *Delete Everything*, and enter your password. From
  Terminal: `sudo /usr/local/blue-onyx-prism/scripts/uninstall_launchd.sh` (add `--purge` to also
  delete the data, logs and the `_blueonyx` account). macOS may not let the uninstaller app delete
  the hidden `_blueonyx` account; it then offers *Remove in Terminal*, where you enter your password
  once more (the account is harmless if you leave it, and a reinstall reuses it).

**Tray / menu bar icon (Windows and macOS).** The Windows and macOS installers also install a small
icon app that starts at login (Windows: system tray, macOS: menu bar) and shows whether the service is
running. Its menu has *Start*, *Stop* and *Restart* (each asks for administrator rights: UAC on Windows,
your password on macOS), *Open Web UI*, and *Quit Tray*, which only closes the icon and leaves the
service running. To bring it back after quitting, open *Blue Onyx Prism* from the Start Menu (Windows)
or from *Applications* (macOS). Linux has no tray; use `systemctl`.

Every installer has a `.sha256` file and a build attestation (`gh attestation verify <file> --repo
patrickgods1/blue-onyx-prism`). The archives below remain the portable option.

## Quick start

Download the archive for your OS from the [releases page](https://github.com/patrickgods1/blue-onyx-prism/releases)
(`blue-onyx-prism-<version>-<os>-<arch>.zip|tar.gz`, with a `.sha256` file). It already contains the
OpenVINO runtime in `openvino/`, the default ONNX Runtime in `onnxruntime/` (CPU on Linux, CoreML on
macOS, DirectML on Windows), the binaries and the helper scripts. Or build from source (see
[Developer setup](#developer-setup)) and run `blue-onyx-prism setup-openvino`. For NVIDIA, DirectML or
CoreML devices run `blue-onyx-prism setup-onnxruntime` (`--flavor auto|cpu|cuda|directml`, default `auto`;
installs into `onnxruntime/` next to the executable, `--dir` to change).

Windows (PowerShell):

```powershell
Expand-Archive blue-onyx-prism-*-windows-x86_64.zip .; cd blue-onyx-prism-*-windows-x86_64
.\blue-onyx-prism.exe download-models --name IPcam-general --add-to-config
.\blue-onyx-prism.exe            # device "auto" (best GPU found); add --force-cpu to use the CPU for this run
```

Linux / macOS:

```sh
tar xzf blue-onyx-prism-*-linux-x86_64.tar.gz && cd blue-onyx-prism-*-linux-x86_64
./blue-onyx-prism download-models --name IPcam-general --add-to-config
./blue-onyx-prism                 # device "auto": CoreML on Apple silicon, Intel GPU or CUDA where present, else CPU
```

Check it: open `http://127.0.0.1:32168/`, or `./test-blue-onyx-prism` (see [Test client](#test-client)).
`--model <path> --family yolo5|yolo26|yolo8|rtdetr|detr|rfdetr|auto` enables that model (adding it to the config when it is not there yet).
`blue-onyx-prism --help` lists every option; `list-models` shows the downloadable catalog.

The first GPU start compiles each model (20-60 s on an iGPU) and stores the result in `cache/`;
later starts take seconds. The HTTP server is up immediately and answers `Model initializing` until
the model is ready.

## Configuration

`blue_onyx_prism_config.json` next to the executable (`--config <file>` to use another). A CLI
flag overrides the file and the merged result is written back when it differs from the file (an
invalid value is an error and the file is left untouched). Three switches are one-off flags:
`--allow-remote-admin`, `--force-cpu` and `--save-ref-image` apply to that run only and are never
written to the file (set `allow_remote_admin`, `force_cpu` or `save_ref_image` there, or on the
Config page, to keep them). Relative paths in the file resolve
against the executable directory. The Windows service reads `blue_onyx_prism_config_service.json`
(created with debug logging on first start).

"Applies" says when a change takes effect: *at once* (on save, from the Config page or an edit of
the file), *model reload* (on save, only that model is reloaded), *restart* (the Config page's
**Restart server**, which reloads the config in the same process, or a process restart) or *process
restart* (stop and start the program or service). "UI" says whether the Config page has a field for
it; every key can also be set in the file.

| Field | Default | Meaning |
|---|---|---|
| `port` | `32168` | HTTP listen port. Integer 1-65535. Applies: process restart. UI: yes |
| `bind_address` | `0.0.0.0` | Address to listen on; `127.0.0.1` = this machine only. IP address. Applies: process restart. UI: yes. See [Remote access](#remote-access) |
| `allow_remote_admin` | `false` | Serve the admin pages and actions to other machines too. Boolean. Applies: at once. UI: yes. See [Remote access](#remote-access) |
| `request_timeout_secs` | `15` | Max time a request may wait in the queue plus processing (the only request timeout). Integer 1-3600. Applies: at once. UI: yes |
| `worker_queue_size` | `0` | Per-model queue length; 0 = auto from timeout and measured inference time. Integer 0-1024. Applies: restart. UI: yes |
| `device` | `"auto"` | Device spec (`auto`, `openvino:gpu`, `ort:cuda`, ..., or legacy `GPU`/`CPU`); see [Hardware and device selection](#hardware-and-device-selection). Falls back to CPU when the device cannot run. Applies: restart. UI: yes |
| `gpu_index` | `0` | GPU to use when several are present. Integer 0-64. Applies: restart. UI: yes |
| `force_cpu` | `false` | Always use the CPU (overrides every per-model device). Boolean. Applies: restart. UI: yes |
| `cache_dir` | `"cache"` | Compiled-model cache; `""` disables it. Path. Applies: restart. UI: yes |
| `openvino_dir` | `null` | OpenVINO runtime dir; default `<exe_dir>/openvino`, else the system install. Path. Applies: restart. UI: no |
| `onnxruntime_dir` | `null` | ONNX Runtime dir (user-managed, never replaced); lookup order: this field, `ORT_DYLIB_PATH`, `onnxruntime/<flavor>/` named by `onnxruntime/active.txt`, `onnxruntime/` itself (older installs), any `onnxruntime/<flavor>/`. Path. Applies: process restart (a process loads one ONNX Runtime). UI: no |
| `auto_download` | `true` | Download missing runtimes and models at startup; see [First run](#first-run). Boolean. Applies: restart. UI: yes |
| `allow_large_downloads` | `false` | Allow downloads over 500 MB (NVIDIA CUDA libraries). Boolean. Applies: restart. UI: yes |
| `download_dir` | `null` | Root for downloaded runtimes and relative model paths (default: the executable's directory). Path. Applies: restart. UI: no |
| `confidence_threshold` | `0.5` | Default minimum confidence (a request's `min_confidence` > 0 overrides it). Number 0-1. Applies: at once. UI: yes |
| `nms_iou` | `0.5` | IoU threshold for NMS-based families. Number 0-1. Applies: at once. UI: yes |
| `object_filter` | `[]` | Only report these labels (case-insensitive); empty = all. List of strings. Applies: at once. UI: yes |
| `night_confidence_threshold` | `null` | Minimum confidence on night / IR frames (dark: mean luma < 50; IR: monochrome, also tinted); raises, never lowers, the threshold. `null` = off. Number 0-1. Applies: at once. UI: yes (Night / IR) |
| `label_thresholds` | `{}` | Minimum confidence per label, e.g. `{"person": 0.6}` (case-insensitive). Numbers 0-1. Applies: at once. UI: Models JSON / config file |
| `night_label_thresholds` | `{}` | Minimum confidence per label on night / IR frames. Numbers 0-1. Applies: at once. UI: yes for `person` (Night / IR) |
| `person_aspect` | `null` | Allowed width / height of a `person` box, `[min, max]` (suggested `[0.2, 0.9]`); `null` = off. Applies: at once. UI: yes (Night / IR) |
| `min_box_area_frac` | `null` | Drop boxes smaller than this share of the image area; `null` = off. Number 0-1. Applies: at once. UI: config file |
| `suppress_overlaps` | `[]` | Drop a box that overlaps another label's box: `[{"drop": "person", "if_overlaps": "potted plant", "iou": 0.5, "night_only": false}]`; fires at IoU >= `iou` or when the other box covers 80% of it. Applies: at once. UI: yes for person / potted plant (Night / IR) |
| `static_suppression` | off | Suppress a box seen at the same place in many requests of one camera (a rock or plant read as a person): `{"enabled": false, "iou": 0.85, "hits": 5, "window_minutes": 30, "labels": ["person"]}`; a request without the box releases it. The camera is the image file name without its timestamp, else the model and frame size. A person standing perfectly still is suppressed too. Applies: at once. UI: yes (on/off, Night / IR) |
| `preprocess` | `null` | Image enhancement before the model's resize (after the night / IR test, which always sees the original frame): `"none"`, `"grayscale"` (BT.601 luma in all channels), `"gamma:<g>"` (`255 (v/255)^g`; below 1 brightens shadows, e.g. `"gamma:0.6"`), `"clahe"` / `"clahe:<clip>,<tiles>"` (CLAHE on luma, default clip 2 and 8 x 8 tiles; colour frames keep their hue). Off by default: enhancement helps some models and hurts others, so compare with `benchmark --variants none,pre:clahe,pre:gamma:0.6` first. Costs a few ms per frame on the preprocess pool. Applies: at once. UI: Models JSON / config file |
| `night_preprocess` | `null` | Enhancement on night / IR frames only (same values as `preprocess`), replacing `preprocess` there; `null` = `preprocess`. Compare with `benchmark --variants night-pre:<mode>`. Applies: at once. UI: Models JSON / config file |
| `verify` | `null` | Verifier cascade for every model without its own `verify`: another configured model checks this model's candidate boxes and a candidate it does not confirm is dropped. `{"enabled": true, "model": "<name>", "labels": ["person"], "min_confidence": 0.5, "iou": 0.3, "night_only": true, "mode": "crop", "pad": 0.2, "max_crops": 4, "aliases": {}}`. Mode `crop`: each candidate (the `max_crops` most confident with a label in `labels`, on night / IR frames when `night_only`) is padded by `pad` x its width / height per side, cropped from the original frame and run on the verifier; it is kept when the verifier reports the same label (case-insensitive; `aliases` maps a verifier label to this model's, e.g. `{"vehicle": "car"}`) with confidence >= `min_confidence` and IoU >= `iou` with the candidate's box inside the crop. Mode `frame`: the verifier runs once on the whole frame and candidates are matched by IoU. The verifier reports its raw detections for these checks (its own night rules, object filter and `verify` do not apply) and never verifies itself. Fail-open: a verifier that is not loaded, busy, past the request timeout or failing leaves the candidates in (`verify_total{outcome="unverified"}`, a throttled warning); rejected ones count in `detections_suppressed_total{reason="verify"}`. Verification time is part of `processMs`; a verified reply may overtake or follow the model's other replies (other requests never wait for it). Compare with `benchmark --variants verify:<model>`. Applies: at once. UI: Models JSON / config file |
| `collect` | off | Dataset collector: saves frames Blue Iris sends, each with a sidecar JSON (model, device, request id, camera, scene, raw detections before the rules, suppressed boxes with their reason, the reply's predictions, verifier outcomes), under `<dir>/<scene>/<camera>/<YYYY-MM-DD>/`, for review and training (layout and schema: [docs/DATASET.md](docs/DATASET.md)). Frames stay on this machine. Saving runs on its own thread after the reply; a full queue drops the frame (`dataset_dropped_total{reason="queue_full"}`). Applies: at once. UI: Dataset page |
| `collect.enabled` | `false` | Save frames at all. Applies: at once. UI: Dataset page |
| `collect.dir` | `null` | Dataset folder; `null` = `dataset` under the data folder (`download_dir`, else the exe dir); relative paths resolve there. Applies: at once. UI: Dataset page |
| `collect.buckets` | all on | Per bucket `{"enabled": true, "rate": 1.0, "per_camera_min_interval_s": 0}` for `positive` (reply has detections; interval 30), `empty` (the model found nothing at the threshold; interval 600), `suppressed` (a night / label threshold, aspect, area, overlap or static rule removed a box), `verify_rejected`, `verify_kept`, `near_threshold` (some label's top score, suppressed and just-below-threshold boxes included, within `margin`, default 0.1, of the request's threshold) and `disagreement` (the verifier rejected a box or, in frame mode, found one the model missed; a night-model frame alone is never a disagreement). `rate` is the share of matching frames kept (0-1, sampled by request id); the interval is per camera and bucket (0-604800 s). A frame matching several buckets is saved once and lists them. Applies: at once. UI: Dataset page |
| `collect.max_gb` | `5` | Size cap of the dataset folder, oldest frames deleted first (above 0, at most 10000). Applies: at once. UI: Dataset page |
| `collect.max_age_days` | `90` | Frames older than this are deleted, 0 = keep (0-3650). Applies: at once. UI: Dataset page |
| `collect.dedup_hamming` | `6` | A frame whose 64-bit dHash is within this many bits of a frame of the same camera saved in the last 10 minutes is dropped as a near duplicate (0-32). Applies: at once. UI: Dataset page |
| `collect.classes` | person, car, bicycle, motorcycle, truck, dog, cat, raccoon, opossum, skunk, fox, deer, squirrel, bird | Class list of the Review page and the label exports (YOLO class ids and COCO category ids follow this order; 1-200 unique names). Reviewed frames record its fingerprint (`classes_version`); a frame with a training label outside the list is left out of exports. Applies: at once. UI: Review page |
| `training` | defaults | Model training ([docs/TRAINING.md](docs/TRAINING.md)): where runs go, the review exports offered, dataset defaults. Read when a run is queued. Applies: at once. UI: Training page |
| `training.dir` | `null` | Folder of the training runs (prepared dataset, `train.log`, held-out benchmark set, `runs.json`); `null` = `training` under the data folder; relative paths resolve there. Applies: at once. UI: config file |
| `training.datasets` | `[]` | Review export folders (YOLO layout) offered on the Training page and used by `train run` without `--dataset`. Applies: at once. UI: Training page |
| `training.epochs` | `50` | Default epochs of a run (1-1000). Applies: at once. UI: Training page |
| `training.negative_share` | `0.15` | Share of frames without objects (hard negatives) kept in the training split; extra empty frames are dropped, too few give a warning (0-0.9). Applies: at once. UI: config file |
| `training.heldout_share` | `0.2` | Share of the review frames held out by camera and day for the benchmark when the export has no split of its own (0-0.9). Applies: at once. UI: config file |
| `training.oversample` | `3` | How often each of your own frames repeats in the training split when the public night subset is mixed in (1-10). Applies: at once. UI: config file |
| `training.timeout_hours` | `48` | A run is stopped after this many hours (1-336). Applies: at once. UI: config file |
| `training.synonyms` | `{}` | Auto-label prompt phrases per class, replacing the built-in ones of that class (`{"raccoon": ["raccoon", "trash panda"]}`; MM-Grounding-DINO and SigLIP prompts; at most 16 phrases, no periods). Applies: at once. UI: config file |
| `log_level` | `"info"` | `trace`, `debug`, `info`, `warn`, `error`. Applies: at once. UI: yes (also the Logs page) |
| `log_timezone` | `"utc"` | Time zone of log timestamps (files, console and the Logs page): `"utc"` (`2026-10-16T00:54:59.580Z`), `"local"` (this computer's zone), an IANA name such as `"America/Los_Angeles"`, or a fixed offset such as `"+08:00"`; others print RFC 3339 with the offset (`2026-10-15T17:54:59.580-07:00`). Applies: at once. UI: yes (a dropdown of UTC, local and common zones; other values set here show as "Custom") |
| `log_path` | `null` | Directory for daily rolling log files `blue_onyx_prism.log.<date>`; the newest 14 are kept (default: stdout only). Path. Applies: process restart. UI: yes |
| `log_format` | `"text"` | Format of the log files under `log_path`: `"text"` or `"json"` (one JSON object per line, for log shippers); the console stays text. Applies: process restart. UI: no |
| `save_image_path` | `null` | Save annotated detections here. Path. Applies: at once. UI: yes |
| `save_ref_image` | `false` | Also save the unannotated image. Boolean. Applies: at once. UI: yes |
| `save_image_max_age_days` | `30` | Delete saved images (`.jpg`/`.jpeg`/`.png` directly in `save_image_path`) older than this many days, 0 = keep; checked every 10 minutes. Integer 0-3650. Applies: at once. UI: yes |
| `save_image_max_mb` | `2048` | Keep the saved-image folder under this many MB, deleting the oldest first, 0 = unlimited. Integer 0-1000000. Applies: at once. UI: yes |
| `max_image_pixels` | `50000000` | Largest image (width x height) accepted for decoding; checked from the header before allocating. Integer 1000-1000000000. Applies: at once. UI: no |
| `intra_threads` | `0` | CPU inference threads: 0 = all physical cores but one on machines with at least 4 cores (the OS, the HTTP threads and the desktop keep a core), every core on smaller ones; set it to use every core. Server and benchmark alike. Integer 0-256. Applies: restart. UI: yes |
| `preprocess_threads` | `0` | Decode + preprocess threads shared by every model; 0 = from the hardware profile (half the cores, 1-8; at most 2 when every model runs on the CPU). Integer 0-256. Applies: restart. UI: no |
| `inflight` | `0` | Concurrent inferences per model; 0 = from the hardware profile and the model's device. A model's own `inflight` wins. Integer 0-64. Applies: restart. UI: no |
| `ort_sessions` | `0` | ONNX Runtime sessions per model, one inference in flight each, each holding its own copy of the model; 0 = from the hardware profile (2 with a discrete GPU of at least 4 GB and 8 GB of memory, so the GPU overlaps one request's compute with the other's copy and NMS; else 1). Integer 0-16. Applies: restart. UI: no |
| `hardware_profile` | `"auto"` | Hardware class the `0` defaults above come from: `auto` (classified from cores, RAM and GPUs) or forced `low`, `mid`, `high`. Applies: restart. UI: no |
| `models_dir` | `"models"` | Where `download-models` and exports put files. Path, must not be empty. Applies: restart. UI: yes |
| `default_model` | `null` | Model serving `/v1/vision/detection`; default = first enabled entry. Model name. Applies: at once. UI: yes (Models card) |
| `models` | `[]` | Model list; per-entry keys in [Multiple models](#multiple-models). Applies: per key, see there. UI: yes (Models card, Models JSON) |
| `benchmark` | see [Per-model devices and benchmarking](#per-model-devices-and-benchmarking) | Benchmark defaults (`benchmark.*` keys, listed there). Applies: at once (read when a benchmark starts). UI: yes (Benchmark page, **Save as default**) |

## Blue Iris setup

1. Settings -> AI: enable the AI server, choose CodeProject.AI Server, address `127.0.0.1` (or the
   host running this service), port `32168`. (Older versions: the "Use AI server on IP/port" fields.)
2. Per camera: Camera settings -> Alerts -> Artificial Intelligence. Leave the custom models field
   empty to use the default object detection (`/v1/vision/detection`, served by `default_model`), or
   enter the configured model names comma separated, e.g. `ipcam-general,yolo26s`
   (each is served via `/v1/vision/custom/<name>`; matching is case-insensitive and ignores a file extension).
3. Click "Trigger now" on the camera (or use the AI test in Blue Iris); the alert shows the label.
4. Check `http://127.0.0.1:32168/stats`: per-model request counters rise and `dropped` should stay 0.
   If it does not, lower the number of models on the GPU or move one to `"device": "CPU"`.

## Developer setup

### Prerequisites

| | Windows | Linux (x86_64 / aarch64) | macOS (Apple silicon) |
|---|---|---|---|
| Rust | `winget install Rustlang.Rustup` (stable, MSVC) | `curl https://sh.rustup.rs -sSf \| sh` | `curl https://sh.rustup.rs -sSf \| sh` |
| C/C++ toolchain | Visual Studio 2022 Build Tools with the "Desktop development with C++" workload (`winget install Microsoft.VisualStudio.2022.BuildTools --override "--wait --quiet --add Microsoft.VisualStudio.Workload.VCTools --includeRecommended"`) | `sudo apt install build-essential pkg-config` | `xcode-select --install` |
| Intel GPU (optional) | Intel graphics driver (OpenCL comes with it) | `sudo apt install intel-opencl-icd` and access to `/dev/dri` (`render` group) | not available (CPU only) |
| Python 3.10-3.12 (only for exporting models by hand; the Config page export brings its own) | `winget install astral-sh.uv` then `uv venv --python 3.11 .venv` | `uv venv --python 3.11 .venv` or `python3 -m venv .venv` | same |

The pinned Rust channel is in `rust-toolchain.toml`; `rustup` picks it up automatically. No `build.rs`,
no system OpenVINO install is needed to build: the `openvino` crate loads the runtime at run time.

### Build and run

```sh
git clone https://github.com/patrickgods1/blue-onyx-prism.git
cd blue-onyx-prism
cargo build --release

# 1. Fetch the OpenVINO runtime (~100 MB) into ./target/release/openvino (next to the exe).
cargo run --release -- setup-openvino
# 1b. Optional, for NVIDIA / DirectML / CoreML: ONNX Runtime into ./target/release/onnxruntime.
cargo run --release -- setup-onnxruntime        # --flavor auto|cpu|cuda|directml
cargo run --release -- list-devices             # what can run here and what `auto` picks

# 2. Get a model. Either download a ready-made ONNX model from Hugging Face ...
cargo run --release -- download-models --name IPcam-general
# ... or export a model (YOLO26, YOLO11, DEIMv2, ...) to OpenVINO IR by hand (needs the Python env, see below).

# 3. Run the server (CPU) and test it.
cargo run --release -- --model models/IPcam-general.onnx --family yolo5 --force-cpu
curl -F image=@test.jpg -F min_confidence=0.4 http://127.0.0.1:32168/v1/vision/detection
```

Drop `--force-cpu` to let `--device auto` (the default) pick the best device, e.g. the Intel GPU. The first GPU start compiles the
model (20-60 s on an iGPU); later starts load from `cache/`.

Development loop:

```sh
cargo test                                  # unit + synthetic post-processing tests, no OpenVINO needed
cargo build --no-default-features           # OpenVINO-only build without ONNX Runtime
cargo clippy --all-targets -- -D warnings
cargo fmt --all
```

### Exporting models

**On demand (no repo or Python needed).** On the Config page, *Resources*, section
*Models · exported on this machine*: click *Export* next to a model. The confirmation names the
download, what runs and the license. Exportable (`model:<name>`):

| Models | Recipe | Family | License |
|---|---|---|---|
| `yolov8{n,s,m,l,x}`, `yolov9{t,s,m,c,e}`, `yolo11{n,s,m,l,x}`, `yolo12{n,s,m,l,x}` | Ultralytics exporter, official `.pt` (assets v8.4.0) | `yolo8` (NMS in the service) | AGPL-3.0 |
| `yolov10{n,s,m,b,l,x}`, `yolo26{n,s,m,l,x}` | same | `yolo26` (end-to-end `[1,300,6]`) | AGPL-3.0 |
| `rfdetr-large` (RF-DETR L, 704x704) | transformers `RfDetrForObjectDetection` + `torch.onnx`, `Roboflow/rf-detr-large` safetensors | `rfdetr` | Apache-2.0 |
| `deimv2-{atto,femto,pico,n,s,m,l,x}` (320 / 416 / 640 px) | official DEIMv2 code (commit `add5bcd`) + `torch.onnx`, `Intellindust/DEIMv2_*_COCO` safetensors | `rtdetr` | Non-commercial / personal use (see below) |
| `mdv6-rtdetr-c` (MegaDetector v6 RT-DETR compact, 640 px; `animal`, `person`, `vehicle`) | RT-DETRv2 modules of the same DEIMv2 code + `torch.onnx`, Microsoft's `MDV6-apa-rtdetr-c.pth` (Zenodo 15398270, 322 MB with training state) | `rtdetr` | Apache-2.0 |
| `mdv6-yolov9-c`, `mdv6-yolov10-c` (MegaDetector v6, 640 px; `animal`, `person`, `vehicle`) | Ultralytics exporter, Microsoft's `.pt` (Zenodo 15398270) | `yolo8` / `yolo26` | AGPL-3.0 |
| `exdark-rfdetr-small`, `exdark-rfdetr-nano` (RF-DETR fine-tuned on ExDark, 512 / 384 px, 12 classes) | transformers `RfDetrForObjectDetection` + `torch.onnx`, `dronefreak/exdark-rfdetr-*` rfdetr checkpoints | `rfdetr` | Non-commercial / personal use (ExDark) |

MegaDetector (camera-trap animal / person / vehicle detector) and the ExDark fine-tunes are night-model
candidates; see [docs/NIGHT_IR.md](docs/NIGHT_IR.md) for what they score on the night sets.
MegaDetector's `vehicle` is its own class, not COCO's `car`.

The first export sets up a private toolchain (about 375 MB download and 1.3 GB on disk on macOS
arm64, up to ~495 MB / 1.7 GB on Linux x86_64): the pinned [uv](https://github.com/astral-sh/uv)
release (SHA-256 checked), Python 3.11 installed by uv, and the hashed package locks in
`scripts/model-export/locks/` (Ultralytics, PyTorch CPU, ONNX, transformers; installed with
`--require-hashes --only-binary :all:`). Then it downloads the model's official weights (SHA-256
pinned; DEIMv2 also the SHA-256 pinned source archive, of which only `engine/**/*.py` and the
training configs `configs/**/*.yml` are extracted), runs the embedded `scripts/export_model.py` offline (Ultralytics and Hugging Face
network access and auto-install off), checks the output against the PyTorch model with ONNX
Runtime, and moves `<name>.onnx` + `.yaml` into `models_dir`. The row shows the stage (1/6 ... 6/6)
and a progress bar, the output goes to the Logs page, and a running export can be cancelled; one
export runs at a time. *and add to config* (checked by default) appends the model to `models` when
it is done (enabled when no other model is; restart to load it). Later exports reuse the toolchain
(only the weights are downloaded; the first export took 60 s on an M1, later ones 3-30 s).

Headless: `blue-onyx-prism fetch --resource model:yolo11s --allow-large` (`--allow-large`, or
`allow_large_downloads`, is only needed while the toolchain is not installed).

Where things live: everything is under `<download_dir or exe dir>/tools/`: `uv/`, `python/`,
`export-env/` (the virtual environment), `export-weights/<name>/`, `export-work/`,
`export-config/` (Ultralytics / Hugging Face settings), `deimv2-src/`. *Remove export toolchain* on
the `tool:uv` row (or deleting `tools/`) frees it; exported models stay. Removing an exported row
deletes its `.onnx` / `.yaml`. The YOLO26-only toolchain of earlier versions (`yolo26-env/` ...) is
replaced by the first export.

This is the one feature that runs downloaded code (everything else is data loaded by whitelist); it
only runs when you click Export or name the resource on the command line. Export subprocesses run
with a minimal environment: `PATH`, `HOME`, the temp dir, locale, `HTTP(S)_PROXY`/`NO_PROXY` and, if
you set them, `SSL_CERT_FILE`/`SSL_CERT_DIR`; nothing else from the service's environment reaches uv
or Python. Cancelling an export kills every helper process it started. Runtime libraries (OpenVINO,
ONNX Runtime, CUDA) are loaded only from the configured runtime directories (`openvino_dir`,
`onnxruntime_dir`, `<download root>/openvino`, `onnxruntime/<flavor>`, `onnxruntime/cuda-libs`) or
the OS library directories, never from the working directory; anyone who can write those directories
can run code as the service, so keep them writable by the service account and administrators only. Supported where PyTorch
2.14 has CPU wheels: Windows x86_64, Linux x86_64 / aarch64, macOS arm64 (14+).

RF-DETR large and DEIMv2 L/X overflow half precision (NaN or garbage with OpenVINO's f16, the
GPU default and the CPU default on ARM): OpenVINO runs them in f32 (`gpu_precision` still
overrides on GPU), which is slower; prefer the smaller DEIMv2 / RF-DETR models on an iGPU.

DEIMv2 license: **non-commercial / personal use only.** The DEIMv2 repository moved from
Apache-2.0 to the non-commercial DEIMv2 License (2026-08-24), which also covers the model weights.
The export pins the last Apache-2.0 source commit (`add5bcd`) and weights published in 2025-10, but
treat DEIMv2 as personal use (e.g. your own cameras); commercial use needs a license from
Intellindust. The DINOv3-based variants (`deimv2-s/m/l/x`) also carry Meta's DINOv3 License for
the backbone.

**By hand** (also makes OpenVINO IR):

```sh
uv venv --python 3.11 .venv
uv pip install -r scripts/requirements-export.txt         # or .venv/bin/pip install -r ...
.venv/bin/python scripts/export_model.py --name yolo26n yolo26s   # Linux / macOS (Windows: .venv/Scripts/python.exe)
.venv/bin/python scripts/export_model.py --name deimv2-n --weights <dir with model.safetensors + config.json> --source <DEIMv2 checkout>
```

This writes `<name>.{xml,bin,onnx,yaml}` and prints the `"models"` snippet for the config file
(`--help` lists the recipes and options). The output directory is `--out-dir` (default
`target/release/models` when it exists, i.e. next to the release exe, else `models/`); point it at
the `models` directory next to the exe you run (the Config page names it). The files then appear
under *Local models (not in config)* with an *Add to config* button (the `.onnx` is preferred: it
also runs on ONNX Runtime, incl. CoreML). Weights are never committed (YOLO weights are AGPL-3.0);
`models/`, `cache/` and `openvino/` are git-ignored.

A model anywhere on the machine (e.g. a fine-tuned RF-DETR `inference_model.onnx`) can be added by
path: the Config page's *Add a model by path* form, `POST /v1/models/add` (form fields `path`,
`name`, `family`, `classes`) or `blue-onyx-prism add-model --path <file|dir> [--name n]
[--family rfdetr] [--classes c.yaml]`. `path` is a `.onnx` / `.xml` file or a directory holding
exactly one model (a Hugging Face `onnx/` subfolder is searched too); a `<model>.yaml`, or a lone
`classes.yaml` / `data.yaml` / `names.yaml` beside it, supplies the class names. Fine-tuned RF-DETR
needs `--family rfdetr` (`auto` treats DETR ports as D-FINE). Duplicate names or files are refused.

### Converting ONNX models to OpenVINO IR

ONNX models (the YOLOv5 ipcam and RT-DETRv2 catalogs) load directly. If one fails on the ONNX
frontend, or to ship FP16 weights, convert it with the same venv:

```sh
.venv/bin/python scripts/convert_onnx_to_ir.py models/rt-detrv2-s.onnx   # -> models/rt-detrv2-s.{xml,bin}
```

To change the pinned export packages, edit `scripts/requirements-model-export.txt` and regenerate
the locks with `scripts/model-export/gen_locks.sh <path to the pinned uv>`.

### Repository layout

- `src/` Rust crate (see CLAUDE.md for the module map)
- `scripts/` service install and uninstall scripts (Windows PowerShell, systemd, launchd) and the model export script
- `deploy/` systemd unit, launchd plist, docker-compose and the container entrypoint; `Dockerfile` at the root
- `docs/PLAN.md` implementation plan and phase checklist
- `.github/workflows/ci.yml` build, lint, test and a CPU smoke test on Windows, Linux and macOS; `release.yml` builds the per-OS archives, attests their provenance and creates a draft release; `check-urls.yml` checks the pinned download URLs weekly and opens an issue when one fails
- `.claude/` Claude Code project configuration (agent definitions)

## Multiple models

List any number of models in `models[]` of the config file:

```json
{
  "default_model": "yolo26s",
  "models": [
    { "name": "yolo26s", "path": "models/yolo26s.xml", "family": "yolo26" },
    { "path": "models/IPcam-general.onnx", "family": "yolo5", "device": "CPU",
      "confidence_threshold": 0.4, "object_filter": ["person", "car"] },
    { "path": "models/rt-detrv2-s.onnx", "family": "rtdetr", "lazy": true, "gpu_precision": "f32" },
    { "path": "models/ipcam-animal.onnx", "family": "yolo5", "enabled": false }
  ]
}
```

Per-model keys (the Models card on the Config page edits `enabled`, `device`, `confidence_threshold`
and the default model; every key is in the *Models JSON* under it):

| Field | Default | Meaning |
|---|---|---|
| `name` | file stem | Name in `/v1/vision/custom/{name}` and Blue Iris; unique, case-insensitive. Applies: model reload (a rename is a remove plus an add). UI: Models JSON |
| `path` | (required) | `.onnx` or OpenVINO IR `.xml` (with its `.bin`); relative paths resolve against the executable directory (or `download_dir`). Applies: model reload. UI: Models JSON, *Add a model by path* |
| `family` | `"auto"` | `auto`, `yolo26`, `yolo5`, `yolo8`, `rtdetr`, `detr`, `rfdetr` (see [Features](#features); fine-tuned RF-DETR needs `rfdetr`). Applies: model reload. UI: Models JSON, *Add a model by path* |
| `classes` | `null` | YAML with `NAMES:`; default `<stem>.yaml` beside the model, then COCO-80. Path. Applies: model reload. UI: Models JSON, *Add a model by path* |
| `device` | `null` | Device spec for this model (any spec of `list-devices`); `null` = the global `device`. Applies: model reload. UI: yes (Models card) |
| `confidence_threshold` | `null` | Minimum confidence for this model; `null` = the global one. Number 0-1. Applies: at once. UI: yes (Models card) |
| `object_filter` | `null` | Only report these labels; `null` = the global `object_filter`. List of strings. Applies: at once. UI: Models JSON |
| `lazy` | `false` | Compile on first request instead of at startup; requests are answered "initializing" at once until it is ready. Boolean. Applies: model reload. UI: Models JSON |
| `gpu_precision` | `null` | Inference precision on an OpenVINO GPU: `f16` (default) or `f32` (slower; for models that overflow f16). Applies: model reload. UI: Models JSON |
| `enabled` | `true` | Load and serve this model; disabled models stay in the config but are not loaded. Boolean. Applies: model reload (start or stop). UI: yes (Models card *Load*) |
| `inflight` | `null` | Concurrent inferences of this model; `null` or `0` = the global `inflight`. Integer 0-64. Applies: restart. UI: no |
| `night_confidence_threshold`, `person_aspect`, `min_box_area_frac`, `suppress_overlaps`, `static_suppression` | absent | Per-model night / IR rules (see the global keys); absent = the global value, a set value replaces it. Applies: at once. UI: Models JSON |
| `label_thresholds`, `night_label_thresholds` | absent | Per-model label thresholds, merged over the global ones (this model's entries win; `0` turns a global entry off). Applies: at once. UI: Models JSON |
| `preprocess`, `night_preprocess` | absent | Per-model image enhancement (see the global keys); absent = the global value, `"none"` turns a global mode off for this model. Applies: at once. UI: Models JSON |
| `night_model` | absent | Name of another configured model that answers this model's night / IR frames (same scene test as `night_confidence_threshold`): the frame is decoded once and handed to that model's worker, which applies its own threshold, night rules and object filter; the response names its execution provider and timings, otherwise unchanged. Falls back to this model (`night_model_fallback_total{reason}`, a throttled warning) while the night model is not served, not loaded or busy. It must not have a `night_model` itself; it may be disabled (then it always falls back). Applies: at once. UI: yes (Models card *Night model*) |
| `verify` | absent | Per-model verifier cascade (see the global key); absent = the global value, a set value replaces it, `{"enabled": false}` turns a global one off for this model. `model` must name another configured model (it may be disabled: then candidates stay unverified). Two models may verify each other. If a frame goes to the `night_model`, the night model's `verify` applies. Applies: at once. UI: Models JSON |

- Names: `name`, else the file stem. Matching is case-insensitive and a `.onnx`/`.xml` suffix is ignored,
  so `/v1/vision/custom/IPcam-General.onnx` reaches `IPcam-general`.
- Only models with `enabled: true` (the default) are compiled and served. Pick them on the Config
  page: the **Models** card has a *Load* checkbox and a *Default* radio per model; **Save** applies
  the selection at once (only the models that changed are loaded or stopped). Disabled models stay
  listed so switching is a click.
- `default_model` (or `--default-model <name>`) serves `/v1/vision/detection`; default is the first
  enabled entry (also used when `default_model` names a disabled model).
- `GET`/`POST /v1/vision/custom/list` returns the loaded (enabled) model names.
- An unknown model name returns HTTP 200 with `success: false` and an error message.
- Per-model overrides: `device`, `confidence_threshold`, `object_filter`, `lazy`, `gpu_precision`
  and `inflight` (table above).
- `blue-onyx-prism download-models --name IPcam-general --add-to-config` downloads and appends the
  model (family and classes file filled in) to the config, skipping names or paths already present.
  New entries are added disabled when the config already has an enabled model (otherwise the first
  one is enabled), so `download-models --all --add-to-config` does not load every model.
- `--model <path>` always enables that model: an entry whose path resolves to the same file is
  enabled, otherwise the model is appended as a new enabled entry; other entries are left as they are.

Each model gets its own worker thread and compiled copy in memory. Expect a few hundred MB per model
on the iGPU (shared system RAM); use `lazy` for rarely used models and `device: "CPU"` to keep the GPU
free for the primary model.

### Per-model devices and benchmarking

Every model can run on its own device: `"device"` in a model entry (any spec of `list-devices`:
`openvino:cpu`, `openvino:gpu.1`, `ort:coreml`, `ort:cuda`, ...) overrides the global `device`, so
OpenVINO and ONNX Runtime models are served side by side (one ONNX Runtime flavor per process).
*Force CPU* overrides every per-model device. On the Config page the Models card has a **Device**
select per model ("Global (...)" = follow the global device), shows the execution provider in use
and links the latest benchmark recommendation.

Which device suits a model differs per model and machine (on an M1, IPcam-general runs fastest on
`openvino:cpu`, D-FINE on `ort:coreml`). The benchmark measures it: each model on each runnable device
over a set of images, graded for accuracy and speed, with a detection check against the CPU:

```sh
blue-onyx-prism benchmark --all-devices                     # config models x runnable devices x configured datasets
blue-onyx-prism benchmark --all-devices --apply             # ...and write each model's recommended device
blue-onyx-prism benchmark --all-devices --report report.html  # standalone HTML (or .md) report
blue-onyx-prism benchmark --list-datasets
```

Or open **Benchmark** in the web UI: choose datasets, models, devices and settings, **Start**
(runs in the background, one run at a time, cancelable, live progress and ETA), then **Use <device>**
per row or **Apply recommended devices to all models** (offers a restart). Benchmarking while serving
competes for the CPU/GPU with live requests. Results go to `benchmark.json` next to the config file;
the CLI and the UI read and write the same file.

- **Datasets** (`benchmark.datasets`, CLI `--dataset`): the built-in, pinned and SHA-256-checked sets
  `coco-cctv` (COCO val2017 CCTV-relevant subset, 180 images), `bmd45-cctv` (real 1080p CCTV,
  vehicles, 44 images) and `exdark-night` (low light, 160 images), downloaded on demand to
  `<data root>/bench/<id>/` (resource `bench:<id>`, also on the Config page's Resources card);
  `sample` (the embedded test image, offline); and folders (`dir:<path>` or
  `{"dir": "...", "gt": "...", "name": "..."}`, CLI `--images <dir> [--gt <file|dir>]`) with optional
  ground truth as COCO instances JSON or YOLO txt labels and tags in `manifest.json`. Without ground
  truth a dataset is scored against a reference model (the most accurate installed one, or
  `benchmark.reference_model`); such scores are marked *relative*.
- **Accuracy**: AP@0.5, AP@[.5:.95] (COCO 101-point), precision/recall/F1 at the model's confidence
  threshold, recall by object size, by dataset and by tag (day/night, complexity, resolution, crowd,
  small objects). Scored classes: person, bicycle, car, motorcycle, bus, truck, dog, cat, bird, horse
  as far as the model has them (IPcam `vehicle` = car/truck/bus); AP over all of a COCO model's classes
  is reported too.
- **Frame-level ROC AUC** (what Blue Iris decides: alert on this frame or not). Box-level ROC is
  undefined for detection, so each frame is one sample per scored class: positive when it has a
  non-ignored object of the class, negative when it has none (frames whose only objects of the class
  are ignore regions, and classes a dataset does not annotate, are left out); its score is the top
  confidence of the class in the frame (0 without a detection). AUC is exact (Mann-Whitney U, ties
  count half), per class, macro over classes (the headline) and micro (pooled), with the ROC curve and
  frame TPR / FPR (alert rate on frames with an object / false-alert rate on frames without) at the
  configured and best thresholds. It uses the 0.05+ predictions: lower scores count as 0, which can
  only understate the AUC slightly.
- **Grades**: speed from the full-request p50 (A < 50 ms, B < 100, C < 200, D < 400, else F); accuracy
  by `benchmark.accuracy_metric` / `--accuracy-metric`: `ap50` (default; AP@0.5 blended 20% with
  small-object recall, A >= 0.70, B >= 0.60, C >= 0.50, D >= 0.40) or `roc_auc` (macro frame ROC AUC,
  A >= 0.95, B >= 0.90, C >= 0.80, D >= 0.70; 0.5 is chance); overall = weighted grade points
  (`benchmark.weights`, default 60% accuracy / 40% speed). The metric also drives the ranking and the
  device recommendation's accuracy tie (0.015 AP50, 0.01 AUC); results kept from earlier runs are
  re-graded with the current metric when a run is saved.
- **Recommendation**: the best-graded device whose detections agree with the CPU reference device
  (devices that disagree, e.g. a broken FP16 path, are never recommended); within 5% of the best p50
  the configured device is kept. The page and the report also rank the models ("best model for this
  machine").
- **Confidence threshold**: the same low-threshold (0.05) predictions are scored at every confidence
  threshold, without extra inference: TP/FP/FN, precision, recall, F1, F2 and false positives per
  image, overall, per dataset, per tag (day, night, small objects) and per class. The best threshold
  per model is exact (precision and recall only change at the predictions' confidence values, so
  every distinct confidence is evaluated; filtering 0.05-threshold output at a higher threshold equals
  running at it, NMS included). Among thresholds within 0.002 of the best score the middle of the
  widest range is taken, rounded down to 0.01 (at least 0.05). The objective is
  `benchmark.threshold_objective` / `--threshold-objective`: `f1` (default), `f2` (favor recall: fewer
  missed people), `precision:<p>` (highest recall with precision >= p, e.g. `precision:0.9` for fewer
  false alerts), `recall:<r>` (highest precision with recall >= r), `youden` (highest frame-level
  TPR - FPR, Youden's J, classes pooled) or `fpr:<x>` (highest frame alert rate with at most x of the
  frames without an object alerting, e.g. `fpr:0.05`); ties go to the higher threshold. The
  recommendation text adds P/R (and frame TPR / FPR) at the best threshold.
  The server threshold is what a model reports: Blue Iris may send its own `min_confidence` per
  request (when it does, it replaces the server threshold for that request), and each camera's
  minimum confidence filters the returned objects again on top.

```sh
blue-onyx-prism benchmark --all-devices --apply-threshold    # also write each model's best confidence_threshold
blue-onyx-prism benchmark --apply --apply-threshold --threshold-objective f2
blue-onyx-prism benchmark --threshold-search --tag night --threshold-objective recall:0.9   # stored predictions, no inference
blue-onyx-prism benchmark --threshold-search --dataset exdark-night --class person --iou 0.6 --json
blue-onyx-prism benchmark --all-devices --accuracy-metric roc_auc --threshold-objective youden
blue-onyx-prism benchmark --threshold-search --threshold-objective fpr:0.05   # frame ROC from stored predictions
```

The predictions behind the search are stored in `benchmark-preds.json` next to `benchmark.json`
(`[class, confidence, x_min, y_min, x_max, y_max]` per prediction and image, per model and device). The
Benchmark page's **Confidence search** card runs the same search (models, device, datasets, tags,
classes, objective with target, IoU) in milliseconds and shows P/R/F1(/F2) vs threshold with hover
values, the configured and best thresholds marked, the frame ROC curve (with ROC AUC per class and frame
TPR / FPR at the configured and best thresholds), a summary and a 0.05-step table, **Apply** /
**Apply all** (writes `confidence_threshold`, same restart flow) and CSV copy/download.

The `benchmark` config section holds the defaults (**Save as default** on the page writes it):

| Field | Default | Meaning |
|---|---|---|
| `benchmark.datasets` | `["coco-cctv", "bmd45-cctv", "exdark-night"]` | Datasets to run: built-in ids, `sample`, `dir:<path>` or `{"dir", "gt", "name"}` objects. UI: yes |
| `benchmark.max_images_per_dataset` | `0` | Images per dataset, evenly spread; 0 = all. Integer >= 0. UI: yes |
| `benchmark.devices` | `[]` | Device specs to run; empty = every runnable option. UI: yes |
| `benchmark.models` | `[]` | Model names; empty = the enabled models whose files are present. UI: yes |
| `benchmark.warmup` | `3` | Untimed warm-up runs per model and device. Integer 0-1000. UI: yes |
| `benchmark.repeat_per_image` | `1` | Timed runs per image. Integer 0-1000. UI: yes |
| `benchmark.reference_model` | `null` | Model whose detections score datasets without ground truth; `null` = the most accurate installed one. UI: yes |
| `benchmark.weights` | `{"accuracy": 0.6, "speed": 0.4}` | Weights of the overall grade, each 0-1. UI: yes (accuracy weight) |
| `benchmark.auto_download_datasets` | `true` | Download the selected built-in datasets when a run starts (large ones still need `allow_large_downloads`). Boolean. UI: no |
| `benchmark.low_priority` | `true` | Run benchmarks at below-normal process priority so the desktop stays responsive (Windows: the whole service for the run, restored after; CLI on Linux/macOS: nice 10). `benchmark --no-low-priority` overrides it. Boolean. Applies: at once. UI: yes |
| `benchmark.threshold_objective` | `"f1"` | Best-threshold objective: `f1`, `f2`, `precision:<p>`, `recall:<r>`, `youden`, `fpr:<x>`. UI: yes |
| `benchmark.accuracy_metric` | `"ap50"` | `ap50` or `roc_auc`: drives the accuracy grade, ranking and device recommendation. UI: yes |

All of them apply at once: they are read when a benchmark starts.

```json
"benchmark": {
  "datasets": ["coco-cctv", "bmd45-cctv", "exdark-night"],
  "max_images_per_dataset": 0, "devices": [], "models": [],
  "warmup": 3, "repeat_per_image": 1, "reference_model": null,
  "weights": {"accuracy": 0.6, "speed": 0.4}, "auto_download_datasets": true,
  "threshold_objective": "f1", "accuracy_metric": "ap50"
}
```

## Web UI

Open `http://<host>:32168/` in a browser:

| Page | What it does |
|---|---|
| `/` | Models (state badge with download progress or failure reason, device, requests, queue), the execution providers *in use*, the `auto` pick, runtimes and GPUs, uptime, API usage; warns when *Force CPU* overrides the device; shows a hint when a newer release exists |
| `/stats` | Per-model state, runtime, device, CPU fallback, requests, dropped, queue, inference/process/round-trip avg/min/max; updates in place every 5 s from `/stats.json` |
| `/test` | Pick, drop or paste an image, choose a model and `min_confidence`; the page posts to `/v1/vision/custom/{model}` and draws the boxes client-side, with a detections table and the raw JSON (same code path and metrics as the API; without JavaScript the form posts to `/test` and the server draws the image) |
| `/config` | Models card (which models load, the default model, device and threshold per model; **Save** applies at once), Resources, server/inference/logging settings with "applies now" / "needs restart" tags and the `models` list as JSON under *Advanced*; **Restart server** reloads the file and recompiles the enabled models without restarting the process; the listener stays up and the old models answer until the new ones are swapped in (a `port` or `log_path` change needs a process restart) |
| `/benchmark` | Benchmark models x devices x datasets in the background (progress, ETA, cancel); grade cards, model ranking, sortable per-device tables (fastest, recommended and configured marked), breakdowns by dataset/tag/class/size/resolution, per-image drill-down with ground truth and predictions drawn; **Use <device>** / **Apply recommended to all** write per-model devices; best confidence threshold per model with a P/R/F1 vs threshold chart, **Apply threshold** / **Apply all thresholds**; **Confidence search** over the stored predictions (filters, objective, chart with hover values, tables, CSV) |
| `/dataset` | Dataset collector (`collect`): on/off, per-bucket toggles, rates and intervals, the caps (applied at once); frames per scene x bucket, per detected class and per camera, disk use, the newest frames per bucket with their sidecars, and **Send to review** (appends the ticked frames to `review-queue.jsonl`). Collection stops on its own on low disk space or a full disk; **Resume** re-arms it |
| `/review` | Review the collected frames one at a time, most uncertain first (filters: bucket, scene, camera, status; the review queue or every frame): accept / reject / relabel each box, restore wrongly suppressed boxes, draw missed ones, confirm empty frames (keyboard: `a` accept all, `r` reject, digits pick a class, `e` empty, `z` undo, `s` save & next, `n` next); export the reviewed frames as YOLO, COCO, X-AnyLabeling and a benchmark `dir:` set split by camera and day; import X-AnyLabeling edits; edit `collect.classes`. Labels: `<stem>.labels.json` beside each frame ([docs/DATASET.md](docs/DATASET.md#review-labels)) |
| `/training` | Import footage (videos, Blue Iris folders, a URL, pinned MEVA clips) and auto-label dataset frames, with their jobs' progress; train a custom model (needs an NVIDIA GPU or Apple silicon): what this machine can train and why not, review export folders, family x size picker with VRAM / time estimates and hardware fit, night augmentation, a queue of runs one at a time with live progress and cancel, the trained models with their held-out benchmark and **Default model** / **night model of** / **verifier of** actions ([docs/TRAINING.md](docs/TRAINING.md)) |
| `/logs` | The last 2,000 log events of this process, live (polls `/logs.json`), with level filter, search, pause/follow, Copy and Download .txt, and the server log level (applies immediately; `POST /config/loglevel` with `level=debug`, form or query) |
| `/prometheus` | Prometheus metrics (`blue_onyx_prism_*{model="..."}`), linked in the footer with the JSON endpoints |

#### Prometheus metrics

Counters and histograms live for the whole process and survive in-process restarts; a model removed
from the config keeps its series. Per model: `blue_onyx_prism_requests_total`,
`blue_onyx_prism_dropped_total{reason="queue_full|timeout|initializing|failed|worker_gone"}`, and the
histograms `blue_onyx_prism_inference_ms`, `blue_onyx_prism_process_ms` and
`blue_onyx_prism_round_trip_ms` (`_bucket{le=...}`, `_sum`, `_count`; bounds 1, 2, 5, 10, 20, 50,
100, 200, 500, 1000, 2000, 5000 ms and `+Inf`). The histograms keep the `_ms` unit in their names
instead of the Prometheus-conventional `_seconds`, for compatibility with existing blue-onyx
dashboards; the `_avg`/`_min`/`_max` gauges are still exported. Process level:
`blue_onyx_prism_restarts_total` and `blue_onyx_prism_requests_rejected_total{reason="overloaded"}`
(detections answered 503 at once because more requests were in flight than every model's queue
can hold plus 16; `stats.json` shows the same as `rejectedOverloaded` and `inflightRequests`).
Other `requests_rejected_total` reasons count detection requests refused before they reach a
model (`body_too_large`, `decode_error`, `bad_request`). Gauges: `model_info`, `model_ready`,
`queue_length`, `queue_capacity`, `inflight_requests`. (`rejected_initializing_total` is now
`dropped_total{reason="initializing"}`.) Process and build:

- `blue_onyx_prism_build_info{version,git_sha,build_date}`: always 1; the labels identify the binary (the older `info{version}` series stays).
- `blue_onyx_prism_download_bytes_total`: bytes downloaded by the resource manager (runtimes, models, datasets).
- `blue_onyx_prism_download_failures_total`: downloads that failed (network, size or SHA-256 mismatch).
- `blue_onyx_prism_process_resident_memory_bytes`: resident memory of the process (omitted where the platform has no cheap source).
- `blue_onyx_prism_process_open_fds`: open file descriptors (Linux only).
- `blue_onyx_prism_dataset_frames_total{bucket}`, `blue_onyx_prism_dataset_dropped_total{reason="queue_full|rate|dup|disk|error"}`, `blue_onyx_prism_dataset_bytes` and `blue_onyx_prism_dataset_collect_disabled`: the dataset collector (`collect`).

`log_path`, `port` and `bind_address` changes need a full process restart. The UI has no
authentication: do not expose the port beyond your LAN.

### Remote access

Blue Iris may run on another machine, so the service listens on every interface by default
(`bind_address` `0.0.0.0`, plus `[::1]` for `localhost`). Administration is answered for clients on
this machine only: from another machine the admin pages and actions (`/config`, `/test`, `/logs`,
`/stats`, `/benchmark`, `/v1/resources*`, `/v1/models/add`, `/v1/fs/browse`, `/v1/benchmark*`,
`/config/*`) return **403** with a short page (browsers) or JSON. Open to every client: the
detection API (`/v1/vision/*`), `/v1/status`, `/v1/status/updateavailable`, the home page `/`, `/stats.json`,
`/prometheus`, `/v1/devices`, `/v1/config` (read-only) and the static files.

- To administer from another machine, set `"allow_remote_admin": true` (or check *Allow remote
  administration* on the Config page from this machine). It applies at once. `--allow-remote-admin`
  does the same for one run only: it is not written to the config file, so the next plain start is
  back to loopback-only. Be aware that remote admin opens the Config page's file browser
  (`GET /v1/fs/browse`, which lists folders and model/YAML file names across the whole filesystem)
  and every admin action (config writes, downloads, model export, restarts) to anyone who can
  reach the port, without authentication, so enable it only on a trusted network. The service
  logs a warning at every start while it is on.
- Cross-site requests (CSRF): a web page on another site cannot drive the admin actions through
  your browser. An admin request that changes something (POST, PUT, PATCH, DELETE) is refused with
  403 when the browser marks it `Sec-Fetch-Site: cross-site`, or when its `Origin` (else
  `Referer`) header names another host:port than the request's `Host` (`Origin: null` counts as
  foreign). Requests with none of these headers are allowed, so `curl` and scripts on this machine
  keep working; browsers always send `Origin` on cross-origin POSTs. The detection API is never
  checked.
- A reverse proxy on this machine (nginx, Caddy, Traefik, e.g. for TLS) makes every proxied request
  arrive from `127.0.0.1`, so everyone who can reach the proxy gets the admin pages: it is
  equivalent to `"allow_remote_admin": true`. `X-Forwarded-For` and similar headers are ignored on
  purpose (any client can send them). Restrict the admin paths in the proxy (or proxy only
  `/v1/vision/*`) when it is reachable from other machines.
- To keep everything on this machine (Blue Iris on the same PC), set `"bind_address": "127.0.0.1"`
  (or `--bind-address 127.0.0.1`); it applies when the process restarts.

## Running as a service

Once installed, control the background service with the same binary:
`blue-onyx-prism service status` prints the state (`running`, `stopped`, `starting`, `stopping`,
`not installed`) and exits 0 when it is running, 3 otherwise, so scripts can test it. Starting and
stopping need administrator rights: run `sudo blue-onyx-prism service start|stop|restart` on macOS
and Linux, or `blue-onyx-prism service start|stop|restart` from an administrator terminal on
Windows. The command waits until the service has actually started or stopped (up to about a
minute) and prints the result. It drives the platform's own service manager (the Windows service
`BlueOnyxPrismService`, the launchd daemon `com.blueonyx.prism`, the systemd unit
`blue-onyx-prism`), so `sc`/`services.msc`, `launchctl` and `systemctl` keep working as before.
On macOS `stop` unloads the daemon until the next `start` or reboot; on Linux and Windows the
service's start type is unchanged, so it starts again at boot. On Windows and macOS the tray /
menu bar icon (`blue-onyx-prism-tray`) does the same from a menu.

**Windows** (elevated PowerShell, from the extracted archive):

```powershell
.\scripts\install_service.ps1        # service BlueOnyxPrismService, auto start, auto restart, firewall rule
.\scripts\uninstall_service.ps1
```

The service runs as the virtual account `NT SERVICE\BlueOnyxPrismService`, not LocalSystem. The
script resets the install dir's ACL (Administrators and SYSTEM full control, Users read and
execute) and gives the service Modify only on its data: `models`, `cache`, `tools`, the runtime
dirs `openvino`, `onnxruntime` and `.downloads` (so runtime downloads and updates keep working),
`blue_onyx_prism_config_service.json` and the files it keeps next to it, plus any `download_dir`,
`models_dir`, `cache_dir`, `save_image_path` and `log_path` the service config names. It can add
new files to the install dir but cannot change the exe or the files shipped with it. A
`save_image_path` or `log_path` set later needs the same grant:
`icacls <dir> /grant "NT SERVICE\BlueOnyxPrismService:(OI)(CI)M"`.
If GPU access fails under the virtual account, reinstall with `-Account LocalSystem` (full control of
the machine) or `-Account DOMAIN\user` (you are prompted for the password; the account needs the
"Log on as a service" right). The service token carries the service SID (`sc.exe sidtype
unrestricted`, so the grants above apply under any account) and only `SeChangeNotifyPrivilege`
(`sc.exe privs`: every other privilege is removed, LocalSystem included). The script raises the
machine-wide `ServicesPipeTimeout` to 600000 ms (applies after a reboot; a larger value is kept) so
the first start can compile models; it records the previous value and `uninstall_service.ps1`
restores it. A reinstall fails with a clear message, instead of half-registering the service,
when services.msc or Event Viewer keeps the old one open. Logs go to the Application event log (source
`BlueOnyxPrism`) and, with `log_path`, to files.

**Linux** (systemd):

```sh
sudo cp -r blue-onyx-prism-<version>-linux-x86_64 /opt/blue-onyx-prism
sudo /opt/blue-onyx-prism/scripts/install_systemd.sh      # creates user blueonyx (render, video groups)
journalctl -u blue-onyx-prism -f
sudo /opt/blue-onyx-prism/scripts/uninstall_systemd.sh    # stop and remove the unit (--purge: also data and user)
```

Re-running `install_systemd.sh` after copying a new release over `/opt/blue-onyx-prism` restarts the
service on the new binary.

The program in `/opt/blue-onyx-prism` stays owned by root. The service user `blueonyx` writes only
to `/var/lib/blue-onyx-prism`: the config (`blue_onyx_prism_config.json`), models, cache, downloads,
`benchmark.json` and the model-export tools. The script seeds that config with `download_dir` and
`cache_dir` pointing there and with `openvino_dir` / `onnxruntime_dir` pointing at the bundled
runtimes. Runtimes named that way are never downloaded or replaced by the service; to use another
ONNX Runtime flavor (CUDA), remove `onnxruntime_dir` from the config, and the service downloads
it into `/var/lib/blue-onyx-prism`. An existing `/opt/blue-onyx-prism/blue_onyx_prism_config.json`
and `models/` are copied over on the first install. Run other commands against that config as the
service user, e.g. `sudo -u blueonyx /opt/blue-onyx-prism/blue-onyx-prism --config
/var/lib/blue-onyx-prism/blue_onyx_prism_config.json fetch --for-config`.
The unit is sandboxed (`ProtectSystem=strict`, `ProtectHome`, `PrivateTmp`, `NoNewPrivileges`,
`DevicePolicy=closed` with only `/dev/dri`, a system-call filter); `deploy/blue-onyx-prism.service`
explains each line, which ones to relax for NVIDIA (`DeviceAllow` lines to uncomment) and where to
add a `save_image_path` or `log_path` outside `/var/lib/blue-onyx-prism` (`ReadWritePaths=`).

**macOS** (launchd):

```sh
sudo cp -r blue-onyx-prism-<version>-macos-aarch64 /usr/local/blue-onyx-prism
sudo /usr/local/blue-onyx-prism/scripts/install_launchd.sh   # KeepAlive; runs as _blueonyx
sudo /usr/local/blue-onyx-prism/scripts/uninstall_launchd.sh # unload, remove the daemon and the program dir (--purge: also data, logs, account; --keep-program: keep the program dir)
```

The daemon runs as the hidden role account `_blueonyx` (created by the script, UID below 500). The
program in `/usr/local/blue-onyx-prism` stays owned by root; the daemon writes only to
`/usr/local/var/blue-onyx-prism` (config, models, cache, downloads; seeded like the Linux one) and
`/var/log/blue-onyx-prism`, where `log_path` writes daily files (the newest 14 are kept).
stdout/stderr go to `/var/log/blue-onyx-prism/stdout.log`, rotated by
`/etc/newsyslog.d/blue-onyx-prism.conf`.

**Docker** (Linux, Intel GPU through `/dev/dri`; image based on `openvino/ubuntu24_runtime`):

```sh
cd deploy && mkdir -p cache config
RENDER_GID=$(getent group render | cut -d: -f3) docker compose up -d --build
```

Create `cache` and `config` yourself as above: the container runs as the image's non-root
`openvino` user, and a directory docker creates for a missing bind mount is owned by root. If your
user's uid differs from the image user's (`docker run --rm --entrypoint id blue-onyx-prism:latest`
prints it), either `sudo chown -R <uid>:<gid> deploy/cache deploy/config` or set `user: "<uid>:<gid>"`
(the directories' owner) in `deploy/docker-compose.yml`. The entrypoint checks this and exits with
the uid in the message instead of starting with a config it cannot save.

The compose file runs the container with a read-only root filesystem (`read_only: true`, a tmpfs at
`/tmp`, `no-new-privileges`, all capabilities dropped). The service writes only to its volumes:
`config` (the config file and `benchmark.json`), `cache`, `models` and the named volume `data` at
`/app/data`, which the shipped config names as `download_dir` (web UI downloads, benchmark datasets,
the model-export toolchain). A config created by an older image has no `download_dir`: add
`"download_dir": "/app/data"` to it (the entrypoint warns until you do). The entrypoint needs no
network; the image's default config has `auto_download: false`.

The image contains IPcam-general; the compose file mounts it through the named volume `models`
(at `/app/models`), so it is there on first start. To bring your own models, replace
`models:/app/models` in `deploy/docker-compose.yml` with `./models:/app/models`, put the files in
`deploy/models` and name them in `deploy/config/blue_onyx_prism_config.json` (copied there from
`deploy/docker-config.json` on first start; edit it, then `docker compose restart`).

The compose file publishes the port on the host's loopback only (`127.0.0.1:32168:32168`). Inside
Docker every client, the host's browser included, arrives from the bridge network, never from
loopback, so with the shipped `"allow_remote_admin": false` the admin pages (Config, Test, Logs,
Benchmark) answer 403 and the detection API works:

- Blue Iris on another machine: switch to the commented LAN line (`"32168:32168"`) in
  `deploy/docker-compose.yml`. Detection is then reachable from the LAN; keep
  `allow_remote_admin` false and change settings by editing the config file in `deploy/config`.
- Web UI: set `"allow_remote_admin": true` in that config file. With the default loopback-only
  port only this host reaches it; from your PC use an SSH tunnel
  (`ssh -L 32168:127.0.0.1:32168 <docker host>`, then open `http://127.0.0.1:32168/`). With the
  LAN line, every LAN client gets the admin pages (see [Remote access](#remote-access)): turn it
  on only while you use them.

The container's health check polls `/v1/status` (`docker ps` shows `healthy` once a model is ready).

## HTTP API

| Method | Path | Notes |
|---|---|---|
| POST | `/v1/vision/detection` | multipart `image`, optional `min_confidence`; default model |
| POST | `/v1/vision/custom/{model}` | same, named model |
| POST, GET | `/v1/vision/custom/list` | configured model names |
| POST | `/v1/models/add` | add a custom model by file or directory path to the config (`path`, `name`, `family`, `classes`) |
| GET | `/v1/fs/browse` | read-only file browser behind the Config page's *Browse...* buttons and the Training page's footage import: `?path=<dir>&kind=model`, `classes` or `video`; lists only folders and `.onnx`/`.xml` (or `.yaml`/`.yml`, or `.mp4`/`.mkv`/`.avi`/`.mov`) files of the machine running the service |
| GET | `/v1/status` | readiness for Blue Iris, Docker `HEALTHCHECK` or systemd: `{ready, status: ready/degraded/initializing/failed, models[{name,state,device,detail}], uptimeSecs, generation, version}`; HTTP 200 when ready or degraded (a failed model next to a usable one), 503 while a non-lazy model is still loading or nothing is usable |
| GET | `/v1/status/updateavailable` | GitHub release check |
| GET | `/`, `/stats`, `/stats.json`, `/prometheus` | UI and metrics |
| GET | `/logs`, `/logs.json?after=<seq>&level=<min>` | log page; recent log events newer than `after` at `level` (`trace`..`error`) or more severe, with `last` as the next cursor |
| GET, POST | `/test`, `/config` | test page, config editor |
| POST | `/config/restart`, `/config/loglevel` | reload config, change log level |
| GET | `/v1/devices` | device options (runnable, downloadable with size, unavailable with reason) and the `auto` pick |
| GET | `/v1/resources` | downloadable resources for this platform: state (installed, downloading with %, queued, needed, optional, available, failed with error and retry), size, what they provide, which models wait for them |
| POST | `/v1/resources/download` | form `id` (e.g. `model:ipcam-bird`); large resources also need `confirm_large=1` unless `allow_large_downloads` |
| POST | `/v1/resources/remove` | form `id`; refused (409) while downloading, while the runtime is loaded ("restart required") or while an enabled model uses the files |
| GET | `/v1/dataset` | dataset collector settings, summary (frames by scene x bucket, by detected class x scene (`by_label`), per camera, newest per bucket), review-queue length and why collection stopped, if it did |
| GET | `/dataset/image?path=`, `/dataset/sidecar?path=` | a collected frame or its sidecar (`path` = `<scene>/<camera>/<date>/<stem>.jpg`; anything else is 404) |
| POST | `/dataset/settings`, `/dataset/review`, `/dataset/resume` | the Dataset page's forms: `collect` settings; `frame=<path>` (repeated) to queue for review; re-arm after a disk stop |
| GET | `/v1/review/next?bucket=&scene=&camera=&label=&status=&source=&after=` | the most uncertain frame to review (after `after`), its position, the review counts (reviewed, remaining, skipped, per class) and the cameras and classes for the filter lists; `label` keeps frames with a box of that class (reviewed labels, else the reply's) |
| GET, POST | `/v1/review/labels?frame=<path>` | a frame's label file (saved, else pre-filled from the sidecar) with its size and sidecar summary; POST a label file (JSON, at most 256 KiB) to save it (`unreviewed`, `reviewed` or `skipped`; pending boxes resolved on `reviewed`) |
| POST | `/v1/review/export` | `{formats, out_dir?, holdout_cameras?, holdout_days?, holdout_fraction?, benchmark_split?, include_unreviewed?, labels?, include_empty?}`: export the reviewed frames (only those with a box of `labels` when given, plus confirmed-empty ones unless `include_empty` is false; a label outside `collect.classes` is a 400) (YOLO, COCO, X-AnyLabeling, benchmark set) to a new folder, default `<data root>/training/exports/<UTC time>` |
| POST | `/v1/review/import`, `/v1/review/classes` | `{dir, treat_unchecked_as_reviewed?}`: import X-AnyLabeling files; `{classes: [...]}`: save `collect.classes` |
| GET | `/v1/training` | training capability (accelerator, families and sizes with estimates and fit, reasons), the runs (state, stage, percent, progress, log tail, dataset counts, installed model, benchmark) and the import / auto-label jobs (`tasks`: state, stage, percent, log tail, summary, result) |
| POST | `/training/queue` | the Training page's form: `model` (repeated: `yolo26n`, `rfdetr-nano`, `rt-detrv2-s`, ...), `dataset` (repeated) / `dataset_path`, `scene`, `classes`, `epochs`, `batch`, `imgsz`, `lr`, `night_aug`, `aug_dark`, `aug_noise`, `aug_gray`, `public`, `benchmark_set`, `name`; `confirm_large=1` for the first environment download unless `allow_large_downloads` |
| POST | `/training/import` | the Import footage form: `source` (`path`, `url`, `dataset`) with `path` / `url` / `dataset`, `camera`, `frames`, `select` (`siglip`, `dhash`), `scene`, `max_fps`, `every`, `dedup_window`, `no_balance`, `license`, `confirm_license`, `confirm_large`; queued on the training queue |
| POST | `/training/autolabel` | the Auto-label form (Training and Review pages): `frames` (filter text), `gdino` (`tiny`, `base`, `large`, `off`), `rfdetr`, `vlm` (`2b`, `4b`, `8b`, `off`), `siglip`, `with_model`, `threshold`, `confirm_large` |
| POST | `/training/cancel`, `/training/remove`, `/training/use`, `/training/datasets`, `/training/remove-env` | `id`: cancel a run; forget a finished one; `id` + `role` (`default`, `night`, `verify`) + `target`: use a trained model; `add` / `remove`: a `training.datasets` folder; delete the training environment |
| GET | `/v1/benchmark` | benchmark state, progress (ETA), partial results of a running benchmark, the saved results (`benchmark.json` without per-image details), model ranking, per-model devices |
| POST | `/v1/benchmark` | start (202; 409 while one runs); form fields, all optional with the `benchmark` config as default: `model`, `device`, `dataset` (repeated), `max_images`, `repeat`, `warmup`, `reference_model`, `accuracy_weight` |
| POST | `/v1/benchmark/cancel`, `/v1/benchmark/settings` | cancel; save the form fields as the `benchmark` config defaults |
| POST | `/v1/benchmark/apply` | `model` + `device` (`""`/`global` clears it), or `all=1` for every saved recommendation; `restart=1` restarts |
| POST | `/v1/benchmark/apply-threshold` | `model` + `threshold` (pairs, repeatable), or `all=1` for every saved best threshold; writes `confidence_threshold`; `restart=1` restarts |
| GET, POST | `/v1/benchmark/threshold-search` | GET: searchable models/devices/datasets/tags/classes; POST: `model` (repeated or `all`), `device` (default recommended), `dataset`, `tag`, `class` (repeated), `objective` (`f1`, `f2`, `precision:<p>`, `recall:<r>`, `youden`, `fpr:<x>`), `iou` (0.5): best threshold, P/R/F1/F2/FP per image at best vs configured, exact curve, 0.05 grid, per dataset/class, CSV; 400 when nothing is stored (run a benchmark first) |
| GET | `/v1/benchmark/images?model=&device=`, `/v1/benchmark/image?set=&file=` | per-image drill-down data; an image of the saved results |
| POST | `/v1/resources/add-to-config` | form `id` of a downloaded model, or `local:<file>` for a model file in `models_dir` (family `auto`): append it to `models` |

The detection response is byte-compatible with CodeProject.AI: `success, message, error, predictions
[{x_min, y_min, x_max, y_max, confidence, label}], count, command, moduleId, executionProvider,
canUseGPU, inferenceMs, processMs, analysisRoundTripMs`.

The admin routes (pages and actions) answer clients on this machine only unless
`allow_remote_admin` is set; others get 403 (see [Remote access](#remote-access)).

## Benchmark

```sh
blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --repeat 50 --warmup 5
blue-onyx-prism-benchmark --model models/yolo26s.xml --family yolo26 --device GPU --compare-cpu --json
blue-onyx-prism-benchmark --model models/IPcam-general.onnx --family yolo5 --all-devices
```

`blue-onyx-prism-benchmark` takes the same flags as `blue-onyx-prism benchmark` (see
[Per-model devices and benchmarking](#per-model-devices-and-benchmarking)). `--device` takes any device
spec and `--all-devices` benchmarks every runnable option for each model (grades, agreement,
recommendation, `benchmark.json`). Flags: `--model`, `--family`, `--device`, `--all-devices`, `--apply`,
`--apply-threshold`, `--threshold-objective`, `--threshold-search` (with `--tag`, `--class`, `--search-model`,
`--search-device`, `--iou`),
`--report <file.html|.md>`, `--no-save`, `--force-cpu`, `--image` (one file), `--dataset` (repeatable),
`--images <dir>`, `--gt`, `--max-images`, `--list-datasets`, `--reference-model`, `--accuracy-weight`,
`--repeat` (timed runs per image; 100 for a single image), `--warmup`, `--compare-cpu` (also runs on CPU
and diffs detections), `--cache-dir` (`""` disables), `--threads`, `--classes`, `--min-confidence`,
`--config` (enabled models from the config when no `--model`), `--json`, `-v`.
If GPU latency is not clearly below CPU latency, the GPU was probably not used (check the log) or
the FP32 path is active.

## Training

```sh
blue-onyx-prism train capability                      # can this machine train, and what fits
blue-onyx-prism train estimate --model rfdetr-small --images 2000 --epochs 50
blue-onyx-prism train prepare --dataset <review export> --scene night+ir --out prep   # dataset only
blue-onyx-prism train run --model yolo26n --model rfdetr-nano --dataset <review export> \
    --scene night+ir --night-aug --epochs 50 --allow-large
blue-onyx-prism train import --path D:\BlueIris\Clips --confirm-license --allow-large   # frames into the dataset
blue-onyx-prism train import --dataset meva-sample                                    # pinned MEVA clips (CC BY 4.0)
blue-onyx-prism train autolabel --frames "import=any,limit=200" --allow-large         # pending auto boxes for review
blue-onyx-prism train autolabel --with-model night-yolo26n-20261023-1200              # self-training re-label
blue-onyx-prism train list
blue-onyx-prism train remove-env
```

Fine-tunes a model on the frames reviewed on the Review page (export them in YOLO format), exports a
static ONNX (checked against PyTorch with ONNX Runtime), installs `models/<name>.onnx` + `.yaml` +
`<name>.provenance.json`, adds it to the config (disabled until you enable it or use it) and benchmarks
it on the held-out frames. The pinned training environment (PyTorch CUDA 12.6 / 13.0 or macOS, the
trainers; about 3 GB with CUDA) is set up under `<data root>/tools/train-env` on the first run. Families,
licences, the dataset layout and the MPS results: [docs/TRAINING.md](docs/TRAINING.md). `train import`
puts frames of your videos and image folders (or a URL whose licence you confirm, or the pinned MEVA
sample) into the dataset folder and the Review queue; `train autolabel` proposes boxes for them
(pending, triaged `accept` / `review`; nothing is accepted for you); both run on the training queue.
Sites that list unsecured cameras are refused.

## Releasing

Tagging steps and the manual NVIDIA (CUDA/TensorRT), Intel iGPU and YOLO26 checks are in
[docs/RELEASE_CHECKLIST.md](docs/RELEASE_CHECKLIST.md). Release binaries carry the commit and build
date (`blue-onyx-prism --version`, `build_info` metric). Each archive ships `LICENSE` and
`THIRD_PARTY_LICENSES.md` (OpenVINO, ONNX Runtime, DirectML, the embedded font and the Rust
crates) and has a `.sha256` file and a build-provenance attestation. Verify a download with:

```sh
sha256sum -c blue-onyx-prism-<version>-linux-x86_64.tar.gz.sha256   # macOS: shasum -a 256 -c ...
gh attestation verify blue-onyx-prism-<version>-linux-x86_64.tar.gz --repo patrickgods1/blue-onyx-prism
```

(Windows: compare `Get-FileHash <zip>` with the `.sha256` file, then the same `gh attestation
verify`.)

## Test client

```sh
test-blue-onyx-prism                                   # embedded sample image -> /v1/vision/detection
test-blue-onyx-prism --model ipcam-general --image cam.jpg --min-confidence 0.4
test-blue-onyx-prism --repeat 50 --parallel 4          # latency summary (client, inferenceMs, processMs)
test-blue-onyx-prism --list                            # /v1/vision/custom/list
test-blue-onyx-prism --save out.jpg                    # annotated copy of the image
```

Also `--url` (default `http://127.0.0.1:32168`) and `--interval-ms`. Exits non-zero if any
response has `success: false` or a non-200 status.

## Troubleshooting

- **GPU not used / `executionProvider` says CPU:** the log prints the available devices. On Windows
  install the current Intel graphics driver; on Linux install `intel-opencl-icd` and make sure the service
  user is in the `render` group (`ls -l /dev/dri`). If the GPU still fails the service falls back to
  CPU and reports `OpenVINO CPU (fallback)`. Under Windows services try `-Account` (see above).
- **First start is slow:** GPU kernels are compiled once (20-60 s per model) and cached in `cache/`;
  keep that directory. Windows service starts allow up to 10 minutes.
- **macOS:** only the CPU plugin exists for OpenVINO on Apple silicon. `auto` uses CoreML through ONNX Runtime
  (run `setup-onnxruntime`); set `"device": "openvino:cpu"` if that is slower for your models. The log line
  `CoreML loads a rewritten copy (...)` shows the CoreML graph rewrite (see Hardware and device selection); `BOP_COREML_REWRITE=0`
  turns it off.
- **`ort:*` device shows as unavailable:** run `list-devices` for the reason. Usually `setup-onnxruntime` has not
  been run, or (NVIDIA) CUDA 12 / cuDNN 9 is not installed. A reason ending in `update onnxruntime-cuda ...
  and restart` means the installed CUDA build is older than the catalog's and has no kernels for the card
  (GTX 10-series needs 1.28+): press Update on the Config page's Resources card (or run
  `blue-onyx-prism fetch --resource onnxruntime-cuda`), then restart.
- **A CPU benchmark made the desktop unresponsive:** benchmarks run at below-normal priority
  (`benchmark.low_priority`, `benchmark --no-low-priority` to opt out) and CPU inference keeps one core
  free (`intra_threads` 0); ONNX Runtime's thread pools do not spin-wait between runs (`BOP_ORT_SPINNING=1`
  restores the default spinning).
- **GPU usage stays around 60 % with DirectML or CUDA:** Task Manager's GPU column shows the busiest engine
  and under-reports compute work; look at the *Compute_0* / *3D* engine graphs or `nvidia-smi dmon -s u`.
  The server runs two ONNX Runtime sessions per model on a discrete GPU (`ort_sessions`) so one request's
  compute overlaps the other's copy and NMS; Blue Iris sends one request per trigger, so 100 % is not
  expected, and the benchmark measures latency one request at a time by design.
- **`Unable to find the openvino_c library`:** run `blue-onyx-prism setup-openvino` or set
  `openvino_dir` / `OPENVINO_INSTALL_DIR`. On Windows install the Visual C++ redistributable.
- **Windows error 1053 / `ServicesPipeTimeout`:** the service did not report in time; reboot once after
  installing (the timeout registry value applies at boot).
- **Blue Iris shows `Unknown model`:** the name in the custom models field must match a configured model name.

## License

MIT. Portions derived from [blue-onyx](https://github.com/xnorpx/blue-onyx) (MIT, Marcus Asteborg).
See `LICENSE`; third-party code in the archives and binaries is listed with its licenses in
`THIRD_PARTY_LICENSES.md`. Ultralytics YOLO weights and the Ultralytics exporter are AGPL-3.0 and are not
distributed with this project: the on-demand export downloads the official weights from
Ultralytics' GitHub release on your request and exports them on your machine (see above), as it
does for RF-DETR large (Apache-2.0) and DEIMv2 (non-commercial / personal use, see the license note above). The blue-onyx IPcam models and RT-DETRv2 models are downloaded from their own
Hugging Face repositories under their own licenses.
