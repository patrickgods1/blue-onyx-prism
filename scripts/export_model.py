#!/usr/bin/env python
"""Export detection models to ONNX (and optionally OpenVINO IR) for Blue Onyx Prism.

Recipes (`--recipe`, default: from the model name):

  ultralytics  Ultralytics YOLOv8/v9/v10/11/12/26 from an official `.pt` (AGPL-3.0). Static 640x640,
               nms=False. YOLOv10/YOLO26 give the end-to-end `[1, 300, 6]` output (family `yolo26`),
               the others the raw `[1, 84, 8400]` head (family `yolo8`, NMS in the service).
  rfdetr       RF-DETR (Hugging Face transformers `RfDetrForObjectDetection`, safetensors, Apache-2.0)
               to a static `pixel_values [1,3,S,S]` -> `logits [1,300,91]`, `pred_boxes [1,300,4]`
               graph (family `rfdetr`; ImageNet normalization is done by the service). A weights
               directory with an `rfdetr`-package training checkpoint (`*.pth` + its `config.json`,
               DetectionBench's `exdark-rfdetr-small` / `-nano`) is loaded into the same transformers
               class; its YAML lists the fine-tune's own classes in COCO spelling.
  deimv2       DEIMv2 (Intellindust, PyTorchModelHubMixin safetensors + config.json) with the official
               DEIMv2 code (`--source`, a checkout of github.com/Intellindust-AI-Lab/DEIMv2): deploy
               mode, inputs `images [1,3,S,S]` (RGB 0..1; the DINOv3 variants' ImageNet normalization
               is part of the graph) and `orig_target_sizes [1,2]` (int64), outputs `labels`, `boxes`,
               `scores` (family `rtdetr`, labels are COCO-80 indices).
  rtdetrv2     An RT-DETRv2 training checkpoint (`.pth`, lyuwenyu/RT-DETR layout; MegaDetector v6
               `MDV6-apa-rtdetr-c`, Apache-2.0) built with the RT-DETRv2 modules of the same DEIMv2
               source (`--source`; PResNet, HybridEncoder `rt_detrv2`, RTDETRTransformerv2,
               PostProcessor) and the model's own class list ([`RTDETRV2_MODELS`]): deploy mode,
               `images [1,3,S,S]` (RGB 0..1, no normalization) + `orig_target_sizes [1,2]` ->
               `labels`, `boxes`, `scores` (family `rtdetr`).

MegaDetector v6 (Microsoft, classes `animal`, `person`, `vehicle`): `mdv6-yolov9-c` / `mdv6-yolov10-c`
go through the Ultralytics recipe (AGPL-3.0; 3-class heads `[1, 7, A]` / `[1, 300, 6]`),
`mdv6-rtdetr-c` through `rtdetrv2`. Their YAML lists the model's own three names.

Usage (inside the project venv, see scripts/requirements-export.txt):

    python scripts/export_model.py --name yolo11n                    # downloads yolo11n.pt itself
    python scripts/export_model.py --name yolo26s --weights yolo26s.pt --no-openvino
    python scripts/export_model.py --name yolo26n yolo26s --int8 --data coco128.yaml
    python scripts/export_model.py --name rfdetr-large --weights path/to/rf-detr-large   # dir with
                                                                     # model.safetensors + config.json
    python scripts/export_model.py --name deimv2-s --weights path/to/DEIMv2_DINOv3_S_COCO \\
        --source path/to/DEIMv2
    python scripts/export_model.py --name yolov8n --out-dir C:/BlueOnyx/models
    python scripts/export_model.py --name mdv6-rtdetr-c --weights MDV6-apa-rtdetr-c.pth \\
        --source path/to/DEIMv2

Before an export the service runs `python -X faulthandler -u -I export_model.py --self-check --recipe <r>`,
which only imports the modules the recipe needs and prints their versions (a broken package or a
native library that cannot load fails in seconds, naming the module).

The service runs this same file (embedded in the binary) for the Config page's "Export" button and
`blue-onyx-prism fetch --resource model:<name>`, as `--recipe <r> --name <name> --family <f>
--weights <verified file or dir> --imgsz <n> --no-openvino --out-dir <staging> --work <dir>
[--source <DEIMv2 code>]` inside its own pinned environment, offline: YOLO_OFFLINE=1,
YOLO_AUTOINSTALL=false, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1 (nothing is downloaded or
pip-installed by the exporters).

Output directory (`--out-dir`, alias `--out`): by default `<repo>/target/release/models` when that
directory exists (the release exe reads `<exe dir>/models`), else `<repo>/models`. Point it at the
`models` directory next to the installed exe (the Config page shows the exact path) so the files
appear under "Local models (not in config)" with an "Add to config" button.

Writes `<name>.onnx` (FP32, opset 17, static shapes, a `COREML_CACHE_KEY` from the graph hash so ONNX
Runtime's CoreML cache never serves an older export; skip with --no-onnx), `<name>.xml` + `.bin`
(OpenVINO IR; skip with --no-openvino) and `<name>.yaml` (80 COCO `NAMES:`), checks the output shapes
for the family and prints the config snippet to paste into `blue_onyx_prism_config.json`.

Export locally and do not commit the results (Ultralytics weights are AGPL-3.0).
"""

from __future__ import annotations

import argparse
import importlib
import json
import os
import shutil
import sys
import types
import xml.etree.ElementTree as ET
from pathlib import Path


REPO = Path(__file__).resolve().parent.parent

RECIPES = ("ultralytics", "rfdetr", "deimv2", "rtdetrv2")
FAMILIES = ("yolo26", "yolo8", "rfdetr", "rtdetr")

# COCO-80 names in the order of the 80-class heads (Ultralytics, DEIMv2) and of the service's
# COCO-80 fallback; RF-DETR's 91-wide head is mapped onto them by the service.
COCO80 = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat",
    "traffic light", "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog",
    "horse", "sheep", "cow", "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella",
    "handbag", "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard", "tennis racket", "bottle",
    "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple", "sandwich", "orange",
    "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
    "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
]
# MegaDetector v6 classes (PytorchWildlife `CLASS_NAMES`). `vehicle` is MegaDetector's own class
# (cars, trucks, ATVs, ...), not COCO's `car`.
MDV6_NAMES = ["animal", "person", "vehicle"]
# RT-DETRv2 checkpoints of the `rtdetrv2` recipe: the DEIMv2 training config with the same
# architecture (relative to --source) and the class names of the head. MDV6-apa-rtdetr-c is
# CameraTraps' `rtdetrv2_r18vd_120e_megadetector.yml`: r18vd, 3 decoder layers, 3 classes, 640.
RTDETRV2_MODELS = {
    "mdv6-rtdetr-c": {
        "config": "configs/deim_rtdetrv2/rtdetrv2_r18vd_120e_coco.yml",
        "names": MDV6_NAMES,
        "imgsz": 640,
    },
}
# Class names of RF-DETR fine-tunes in COCO spelling (ExDark: People, Motorbike, Table).
RFDETR_CLASS_TO_COCO = {"people": "person", "motorbike": "motorcycle", "table": "dining table"}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]


def log(msg: str) -> None:
    print(f"[export_model] {msg}", flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"[export_model] error: {msg}")


def default_out_dir() -> Path:
    """`<repo>/target/release/models` when it exists (next to the release exe), else `<repo>/models`."""
    release = REPO / "target" / "release" / "models"
    return release if release.is_dir() else REPO / "models"


def recipe_for(name: str) -> str:
    n = name.lower()
    if n.startswith(("yolo", "mdv6-yolo")):
        return "ultralytics"
    if n.startswith("rfdetr"):
        return "rfdetr"
    if n.startswith("deimv2"):
        return "deimv2"
    if n in RTDETRV2_MODELS:
        return "rtdetrv2"
    fail(f"no recipe for {name!r}; pass --recipe")
    return ""


def family_for(name: str, recipe: str) -> str:
    n = name.lower()
    if recipe == "ultralytics":
        return "yolo26" if n.startswith(("yolov10", "yolo26", "mdv6-yolov10")) else "yolo8"
    return {"rfdetr": "rfdetr", "deimv2": "rtdetr", "rtdetrv2": "rtdetr"}[recipe]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", nargs="+", default=None, help="model name(s): yolov8n, yolo26s, rfdetr-large, deimv2-n, ...")
    p.add_argument("--recipe", choices=RECIPES, default=None, help="default: from the name")
    p.add_argument("--family", choices=FAMILIES, default=None, help="expected family (checks the output shape); default: from the name")
    p.add_argument(
        "--weights",
        type=Path,
        default=None,
        help="local weights of one model: the .pt (ultralytics), the .pth or its directory (rtdetrv2) or the directory "
        "with model.safetensors + config.json (rfdetr, deimv2); nothing is downloaded when given",
    )
    p.add_argument(
        "--source", type=Path, default=None, help="DEIMv2 source checkout (the directory with engine/; deimv2, rtdetrv2)"
    )
    p.add_argument("--imgsz", type=int, default=None, help="square input size (default: 640; RF-DETR L 704; DEIMv2 from config.json)")
    p.add_argument(
        "--out-dir",
        "--out",
        dest="out",
        type=Path,
        default=None,
        help="where to write the model files (default: <repo>/target/release/models if it exists, else <repo>/models)",
    )
    p.add_argument("--int8", action="store_true", help="Ultralytics IR only: INT8 post-training quantization (needs --data)")
    p.add_argument("--data", default="coco128.yaml", help="dataset yaml used for INT8 calibration")
    p.add_argument("--fp32", action="store_true", help="keep FP32 weights in the IR instead of FP16")
    p.add_argument("--opset", type=int, default=17, help="ONNX opset")
    p.add_argument("--no-onnx", action="store_true", help="skip the <name>.onnx export")
    p.add_argument("--no-openvino", action="store_true", help="skip the OpenVINO IR export (ONNX only)")
    p.add_argument("--work", type=Path, default=REPO / ".export_work", help="scratch directory")
    p.add_argument(
        "--self-check",
        action="store_true",
        help="only import what --recipe needs and print the versions (checks the environment; exports nothing)",
    )
    args = p.parse_args()
    if args.self_check:
        if args.recipe is None:
            p.error("--self-check needs --recipe")
        return args
    if not args.name:
        p.error("the following arguments are required: --name")
    if args.out is None:
        args.out = default_out_dir()
    if args.no_onnx and args.no_openvino:
        p.error("--no-onnx and --no-openvino leave nothing to export")
    if len(args.name) != 1 and (args.weights is not None or args.family or args.imgsz):
        p.error("--weights, --family and --imgsz need exactly one --name")
    return args


# ---------------------------------------------------------------------------------------------
# Checks
# ---------------------------------------------------------------------------------------------


def onnx_ports(path: Path) -> tuple[dict, dict]:
    """({input name: shape}, {output name: shape}) of an ONNX file (-1 = dynamic)."""
    import onnx

    graph = onnx.load(str(path), load_external_data=False).graph
    init = {i.name for i in graph.initializer}

    def dims(v):
        return [d.dim_value if d.HasField("dim_value") else -1 for d in v.type.tensor_type.shape.dim]

    return (
        {i.name: dims(i) for i in graph.input if i.name not in init},
        {o.name: dims(o) for o in graph.output},
    )


def check_ports(name: str, family: str, inputs: dict, outputs: dict, imgsz: int, nc: int = len(COCO80)) -> str:
    """Fail unless the ports are what the service's `family` expects (`nc` classes); returns a
    summary."""
    shapes = list(outputs.values())
    if family == "yolo26":
        ok = len(shapes) == 1 and len(shapes[0]) == 3 and shapes[0][0] == 1 and shapes[0][2] == 6
        want = "one output [1, N, 6] (NMS-free)"
    elif family == "yolo8":
        ok = len(shapes) == 1 and shapes[0][:2] == [1, 4 + nc] and shapes[0][2] > 4 + nc
        want = f"one output [1, {4 + nc}, A]"
    elif family == "rfdetr":
        lg, bx = outputs.get("logits"), outputs.get("pred_boxes")
        ok = (
            list(inputs) == ["pixel_values"]
            and inputs["pixel_values"] == [1, 3, imgsz, imgsz]
            and lg is not None and len(lg) == 3 and lg[0] == 1
            and bx is not None and bx == [1, lg[1], 4]
        )
        want = f"pixel_values [1, 3, {imgsz}, {imgsz}] -> logits [1, Q, C], pred_boxes [1, Q, 4]"
    elif family == "rtdetr":
        lb, bx, sc = outputs.get("labels"), outputs.get("boxes"), outputs.get("scores")
        ok = (
            inputs.get("images") == [1, 3, imgsz, imgsz]
            and inputs.get("orig_target_sizes") == [1, 2]
            and lb is not None and len(lb) == 2
            and bx == [1, lb[1], 4]
            and sc == lb
        )
        want = f"images [1, 3, {imgsz}, {imgsz}] + orig_target_sizes [1, 2] -> labels [1, N], boxes [1, N, 4], scores [1, N]"
    else:
        fail(f"unknown family {family}")
        return ""
    summary = f"inputs {inputs}, outputs {outputs}"
    if not ok:
        fail(f"{name}: unexpected ports for family {family}: {summary}; expected {want}")
    return summary


def ir_output_shape(xml_path: Path) -> list[int]:
    root = ET.parse(xml_path).getroot()
    results = [l for l in root.iter("layer") if l.get("type") == "Result"]
    if not results:
        return []
    port = results[0].find("input/port")
    if port is None:
        return []
    return [int(d.text) for d in port.findall("dim")]


def synthetic_image(size: int):
    """A smooth, deterministic RGB 0..1 test image (gradients and blocks, not noise), [1,3,S,S]."""
    import numpy as np

    y, x = np.mgrid[0:size, 0:size].astype(np.float32) / max(size - 1, 1)
    img = np.stack([x, y, 0.5 * (x + y)])
    s = size // 4
    img[:, s : 2 * s, s : 3 * s] = np.array([0.8, 0.2, 0.2], dtype=np.float32)[:, None, None]
    img[:, 2 * s : 3 * s, 2 * s : 3 * s] = np.array([0.1, 0.3, 0.9], dtype=np.float32)[:, None, None]
    return img[None].astype(np.float32)


def export_threads_from_env() -> int | None:
    """`BOP_EXPORT_THREADS` (a positive integer) or None: the training script sets it to cap the
    CPU threads of the export after training (all cores but one); the model export leaves it unset."""
    try:
        n = int(os.environ.get("BOP_EXPORT_THREADS", ""))
    except ValueError:
        return None
    return n if n > 0 else None


def compare_with_onnxruntime(name: str, onnx_path: Path, feeds: dict, torch_out: dict, threads: int | None = None) -> dict:
    """Run the ONNX file in ONNX Runtime and compare with PyTorch's outputs (`name -> array`).
    `threads` (else `BOP_EXPORT_THREADS`, else ONNX Runtime's default: every core) caps the
    session's intra-op threads."""
    import numpy as np
    import onnxruntime as ort

    threads = threads or export_threads_from_env()
    kw = {}
    if threads:
        so = ort.SessionOptions()
        so.intra_op_num_threads = threads
        kw["sess_options"] = so
    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"], **kw)
    got = dict(zip([o.name for o in sess.get_outputs()], sess.run(None, feeds)))
    for k, ref in torch_out.items():
        out = got[k]
        if out.shape != ref.shape:
            fail(f"{name}: ONNX Runtime output {k} has shape {out.shape}, PyTorch {ref.shape}")
        if not np.all(np.isfinite(out.astype(np.float64))):
            fail(f"{name}: ONNX Runtime output {k} has non-finite values")
    log(f"{name}: ONNX Runtime check: " + ", ".join(f"{k} {tuple(v.shape)}" for k, v in got.items()))
    return got


def pin_output_shapes(name: str, path: Path, got: dict) -> None:
    """Write the output shapes ONNX Runtime produced into the graph where the exporter left them
    dynamic (the input is static, so they are too); OpenVINO and CoreML compile static graphs
    better."""
    import onnx

    model = onnx.load(str(path))
    changed = False
    for o in model.graph.output:
        shape = got[o.name].shape
        dims = o.type.tensor_type.shape.dim
        if len(dims) != len(shape):
            fail(f"{name}: output {o.name} has rank {len(dims)} in the graph but {len(shape)} at run time")
        for d, n in zip(dims, shape):
            if not d.HasField("dim_value"):
                d.ClearField("dim_param")
                d.dim_value = int(n)
                changed = True
            elif d.dim_value != n:
                fail(f"{name}: output {o.name} is {list(shape)} at run time, not the graph's shape")
    if changed:
        onnx.save(model, str(path))
        log(f"{name}: pinned the output shapes to the static run-time shapes")


def stamp_coreml_cache_key(path: Path) -> None:
    """Set the `COREML_CACHE_KEY` metadata to a hash of the graph. ONNX Runtime's CoreML provider
    otherwise keys its compiled-model cache by file path, so a re-export (or the export that
    replaces the earlier `rfdetr-large` download) would load the stale compiled model."""
    import hashlib

    import onnx

    model = onnx.load(str(path))
    for i, p in reversed(list(enumerate(model.metadata_props))):
        if p.key == "COREML_CACHE_KEY":
            del model.metadata_props[i]
    key = hashlib.sha256(model.SerializeToString()).hexdigest()[:32]
    model.metadata_props.add(key="COREML_CACHE_KEY", value=key)
    onnx.save(model, str(path))


def write_names(out: Path, name: str, names: list[str], nc: int = len(COCO80)) -> Path:
    """`<name>.yaml` with the `nc` class names of the head, in head order."""
    if len(names) != nc or not all(isinstance(n, str) and n.strip() for n in names):
        fail(f"{name}: {len(names)} class names; expected {nc} (the classes of the model's head)")
    path = out / f"{name}.yaml"
    path.write_text("NAMES:\n" + "".join(f"  - {n}\n" for n in names), encoding="utf-8")
    return path


def onnx_to_ir(onnx_path: Path, out: Path, name: str, fp32: bool) -> Path:
    """OpenVINO IR from the ONNX (needs the `openvino` Python package; by-hand use only)."""
    import openvino as ov

    xml = out / f"{name}.xml"
    ov.save_model(ov.convert_model(str(onnx_path)), str(xml), compress_to_fp16=not fp32)
    return xml


# ---------------------------------------------------------------------------------------------
# Ultralytics
# ---------------------------------------------------------------------------------------------


def ultralytics_model(name: str, args: argparse.Namespace):
    """`YOLO(<work>/<name>.pt)`. With --weights the file is copied there first, so Ultralytics never
    downloads and its outputs land in the work dir, not next to the weights."""
    from ultralytics import YOLO

    # Ultralytics writes `<name>.onnx` / `<name>_openvino_model/` next to the weights and may
    # download `<name>.pt` into the current directory: run inside the work dir.
    os.chdir(args.work)
    weights = args.work / f"{name}.pt"
    if args.weights is not None:
        src = args.weights.resolve()
        if not src.is_file():
            fail(f"--weights {src} does not exist")
        if src != weights:
            shutil.copyfile(src, weights)
    return YOLO(str(weights) if weights.exists() else f"{name}.pt")


def top_detection_scores(a, k: int = 20, nc: int = len(COCO80)):
    """The `k` highest class scores of an Ultralytics detection output, sorted ascending, from
    either layout: end-to-end `[1, N, 6]` (score in column 4) or the raw head `[1, 4 + C, A]`
    (sigmoid class scores in rows 4..). The end-to-end head keeps the top (anchor, class) pairs,
    so both layouts give the same top-k of the same model. None for an unknown layout."""
    import numpy as np

    a = np.asarray(a, dtype=np.float64)
    if a.ndim == 3 and a.shape[0] == 1 and a.shape[2] == 6:
        s = a[0, :, 4]
    elif a.ndim == 3 and a.shape[0] == 1 and a.shape[1] == 4 + nc:
        s = a[0, 4:, :].ravel()
    else:
        return None
    return np.sort(s)[-k:]


def check_ultralytics_numerics(name: str, yolo, onnx: Path, input_name: str, imgsz: int, nc: int = len(COCO80)) -> None:
    """Run the PyTorch model and the exported ONNX (ONNX Runtime, CPU) on the synthetic image and
    fail when their top-20 detection scores differ by more than 1e-2 (as the RF-DETR / DEIMv2
    recipes do). With the raw `[1, 84, A]` head in both, the boxes of the 20 best anchors must also
    agree within 0.5 px. When the PyTorch eval output has another layout than the ONNX graph
    (Ultralytics' export mode can reshape the end-to-end head), only the top scores are compared,
    which both layouts carry."""
    import numpy as np
    import torch

    x = synthetic_image(imgsz)
    model = yolo.model.float().eval()
    got = compare_with_onnxruntime(name, onnx, {input_name: x}, {})
    out = next(iter(got.values()))
    # A YOLOv10 head pickled by an older Ultralytics (MegaDetector's MDV6-yolov10-c, 8.3) has
    # `end2end` False, so PyTorch eval runs its one-to-many head while the exporter writes the
    # one-to-one head ([1, 300, 6]): compare the head that was exported.
    head = model.model[-1] if hasattr(model, "model") else None
    if out.ndim == 3 and out.shape[2] == 6 and getattr(head, "one2one_cv2", None) is not None and not getattr(head, "end2end", True):
        head.end2end = True
        log(f"{name}: comparing the exported one-to-one head (end2end) in PyTorch")
    with torch.no_grad():
        ref = model(torch.from_numpy(x))
    # Eval mode returns (output, extras) in most heads; take the first tensor.
    while isinstance(ref, (tuple, list)) and ref:
        ref = ref[0]
    if isinstance(ref, dict):
        ref = ref.get("one2one", next(iter(ref.values()), None))
        while isinstance(ref, (tuple, list)) and ref:
            ref = ref[0]
    if not isinstance(ref, torch.Tensor):
        fail(f"{name}: the PyTorch model returned {type(ref).__name__}, not a tensor; cannot check the export")
    ref = ref.detach().float().numpy()
    a, b = top_detection_scores(ref, nc=nc), top_detection_scores(out, nc=nc)
    if a is None or b is None:
        fail(f"{name}: cannot compare PyTorch {ref.shape} with ONNX Runtime {out.shape} (unknown output layout)")
    if ref.shape != out.shape:
        log(f"{name}: PyTorch output {ref.shape} vs ONNX {out.shape}: comparing the top-20 scores only")
    diff = float(np.abs(a - b).max())
    log(f"{name}: top-20 score difference PyTorch vs ONNX Runtime: {diff:.2e}")
    if diff > 1e-2:
        fail(f"{name}: ONNX Runtime and PyTorch disagree (top scores differ by {diff:.3f})")
    if ref.shape == out.shape and ref.shape[1] == 4 + nc:
        best = np.argsort(ref[0, 4:, :].max(axis=0))[-20:]
        box_diff = float(np.abs(ref[0, :4, best] - out[0, :4, best]).max())
        log(f"{name}: box difference of the 20 best anchors: {box_diff:.2e} px")
        if box_diff > 0.5:
            fail(f"{name}: ONNX Runtime and PyTorch boxes disagree by {box_diff:.2f} px")


def ultralytics_names(yolo) -> list[str]:
    """The class names of an Ultralytics model, in head order (80 COCO names, or MegaDetector's 3)."""
    names = yolo.names
    return [names[k] for k in sorted(names, key=int)] if isinstance(names, dict) else list(names)


def export_ultralytics(name: str, family: str, imgsz: int, args: argparse.Namespace) -> dict:
    out: dict = {}
    names = ultralytics_names(ultralytics_model(name, args))
    nc = len(names)
    if not args.no_openvino:
        log(f"{name}: exporting OpenVINO IR")
        model = ultralytics_model(name, args)
        kwargs = dict(format="openvino", imgsz=imgsz, dynamic=False, nms=False, batch=1)
        new_kwargs = dict(kwargs)
        if args.int8:
            kwargs.update(int8=True, data=args.data)
            new_kwargs.update(quantize=8, data=args.data)
        elif not args.fp32:
            kwargs.update(half=True)
            new_kwargs.update(quantize=16)
        # Ultralytics >= 8.4 replaced half/int8 with `quantize`; try the new spelling first.
        try:
            res = Path(model.export(**new_kwargs))
        except TypeError:
            res = Path(model.export(**kwargs))
        export_dir = res if res.is_dir() else res.parent
        xml = next(export_dir.glob("*.xml"))
        shape = ir_output_shape(xml)
        check_ports(name, family, {}, {"output0": shape}, imgsz, nc)
        out["xml"] = args.out / f"{name}.xml"
        shutil.copy2(xml, out["xml"])
        shutil.copy2(xml.with_suffix(".bin"), args.out / f"{name}.bin")
    if not args.no_onnx:
        log(f"{name}: exporting ONNX (opset {args.opset})")
        model = ultralytics_model(name, args)
        onnx = Path(model.export(format="onnx", imgsz=imgsz, dynamic=False, nms=False, batch=1, opset=args.opset))
        inputs, outputs = onnx_ports(onnx)
        log(f"{name}: " + check_ports(name, family, inputs, outputs, imgsz, nc))
        check_ultralytics_numerics(name, ultralytics_model(name, args), onnx, list(inputs)[0], imgsz, nc)
        out["onnx"] = args.out / f"{name}.onnx"
        shutil.copy2(onnx, out["onnx"])
        stamp_coreml_cache_key(out["onnx"])
    out["yaml"] = write_names(args.out, name, names, nc)
    out["names"] = names
    return out


# ---------------------------------------------------------------------------------------------
# Hugging Face recipes (RF-DETR, DEIMv2)
# ---------------------------------------------------------------------------------------------


def weights_dir(name: str, args: argparse.Namespace, files: tuple = ("model.safetensors", "config.json")) -> Path:
    if args.weights is None:
        fail(f"{name}: --weights <dir with {' and '.join(files)}> is required (nothing is downloaded)")
    d = args.weights.resolve()
    if d.is_file():
        d = d.parent
    for f in files:
        if not (d / f).is_file():
            fail(f"{name}: {d / f} is missing")
    return d


def torch_export(model, inputs: tuple, input_names: list[str], output_names: list[str], path: Path, opset: int) -> None:
    import torch

    with torch.no_grad():
        # The TorchScript exporter: static shapes, no onnxscript needed.
        torch.onnx.export(
            model,
            inputs,
            str(path),
            input_names=input_names,
            output_names=output_names,
            opset_version=opset,
            do_constant_folding=True,
            dynamo=False,
        )


def rfdetr_from_training_checkpoint(name: str, src: Path, imgsz: int, custom_classes: bool = False):
    """An RF-DETR fine-tune saved by the `rfdetr` package (`checkpoint_best_total.pth` +
    `config.json` with `model_config` and `class_names`, e.g. DetectionBench's ExDark models) in
    transformers' `RfDetrForObjectDetection`: the config is rebuilt from `model_config` the way
    transformers' `convert_rf_detr_weights_to_hf.py` builds it, and transformers maps the original
    state dict onto its modules (`from_pretrained(None, state_dict=...)`; nothing may be missing,
    mismatched or unexpected but the empty keypoint mask). The checkpoint is read with
    `torch.load(weights_only=True)`. Returns (model, class names in head order mapped to COCO
    spellings, number of logits). With `custom_classes` (a model trained by
    `scripts/train_model.py` on its own class list, e.g. `raccoon`) the names are kept as they are
    instead of being mapped to COCO spellings."""
    import torch
    from transformers import RfDetrConfig, RfDetrForObjectDetection

    cfg = json.loads((src / "config.json").read_text(encoding="utf-8"))
    mc = cfg.get("model_config") or {}
    names = cfg.get("class_names") or []
    if mc.get("encoder") != "dinov2_windowed_small":
        fail(f"{name}: unsupported RF-DETR encoder {mc.get('encoder')!r} (only dinov2_windowed_small)")
    if mc.get("projector_scale") != ["P4"] or not mc.get("two_stage") or not mc.get("bbox_reparam"):
        fail(f"{name}: unsupported RF-DETR layout (projector {mc.get('projector_scale')}, two_stage, bbox_reparam)")
    if mc.get("resolution") != imgsz:
        fail(f"{name}: --imgsz {imgsz} but the model was trained at {mc.get('resolution')}")
    if len(names) != mc.get("num_classes"):
        fail(f"{name}: {len(names)} class names for num_classes {mc.get('num_classes')}")
    if custom_classes:
        coco = [str(n).strip() for n in names]
        if not all(coco):
            fail(f"{name}: empty class name in {names}")
    else:
        coco = [RFDETR_CLASS_TO_COCO.get(n.lower(), n.lower()) for n in names]
        unknown = [n for n in coco if n not in COCO80]
        if unknown:
            fail(f"{name}: class names without a COCO spelling: {unknown}")
    backbone = {
        "model_type": "rf_detr_dinov2",
        "attention_probs_dropout_prob": 0.0,
        "drop_path_rate": 0.0,
        "hidden_act": "gelu",
        "hidden_dropout_prob": 0.0,
        "hidden_size": 384,
        # The checkpoint's position embeddings cover `positional_encoding_size` patches per side
        # (RF-DETR base: 37 x 14 = 518 px, trained at 560); the export interpolates them to imgsz.
        "image_size": int(mc.get("positional_encoding_size") or imgsz // mc["patch_size"]) * mc["patch_size"],
        "initializer_range": 0.02,
        "layer_norm_eps": 1e-06,
        "layerscale_value": 1.0,
        "mlp_ratio": 4,
        "num_channels": 3,
        "num_hidden_layers": 12,
        "num_attention_heads": 6,
        "num_windows": mc["num_windows"],
        "out_features": [f"stage{i}" for i in mc["out_feature_indexes"]],
        "patch_size": mc["patch_size"],
        "qkv_bias": True,
        "use_swiglu_ffn": False,
    }
    nl = int(mc["num_classes"]) + 1
    config = RfDetrConfig(
        d_model=mc["hidden_dim"],
        num_queries=mc["num_queries"],
        decoder_self_attention_heads=mc["sa_nheads"],
        decoder_cross_attention_heads=mc["ca_nheads"],
        decoder_n_points=mc["dec_n_points"],
        projector_scale_factors=[1.0],
        intermediate_size=1024,
        decoder_layers=mc["dec_layers"],
        backbone_config=backbone,
        num_labels=nl,
    )
    found = sorted(src.glob("*.pth"))
    if len(found) != 1:
        fail(f"{name}: expected exactly one .pth in {src}, found {len(found)}")
    ck = torch.load(str(found[0]), map_location="cpu", weights_only=True)
    state = ck.get("model") if isinstance(ck, dict) else None
    if not isinstance(state, dict):
        fail(f"{name}: {found[0].name} has no `model` state dict")
    model, info = RfDetrForObjectDetection.from_pretrained(None, config=config, state_dict=state, output_loading_info=True)
    unexpected = [k for k in info["unexpected_keys"] if not k.endswith("_kp_active_mask")]
    if info["missing_keys"] or unexpected or info["mismatched_keys"]:
        fail(
            f"{name}: the checkpoint does not fit RfDetrForObjectDetection (missing {sorted(info['missing_keys'])[:5]}, "
            f"unexpected {unexpected[:5]}, mismatched {info['mismatched_keys'][:5]})"
        )
    log(f"{name}: {found[0].name} (rfdetr {ck.get('rfdetr_version', '?')}, {len(state)} tensors): {nl} logits, classes {coco}")
    return model.eval(), coco, nl


def export_rfdetr(name: str, family: str, imgsz: int, args: argparse.Namespace) -> dict:
    import torch
    from transformers import RfDetrForObjectDetection

    names = None
    wdir = args.weights.resolve() if args.weights is not None else None
    if wdir is not None and wdir.is_dir() and any(wdir.glob("*.pth")):
        # A fine-tune saved by the rfdetr package (DetectionBench ExDark): its own classes.
        src = weights_dir(name, args, ("config.json",))
        log(f"{name}: loading the rfdetr training checkpoint in {src} with transformers")
        model, names, nl = rfdetr_from_training_checkpoint(name, src, imgsz, getattr(args, "custom_classes", False))
    else:
        src = weights_dir(name, args)
        log(f"{name}: loading {src} with transformers")
        model = RfDetrForObjectDetection.from_pretrained(str(src), local_files_only=True).eval()
        labels = model.config.id2label
        if len(labels) != 91 or labels.get(1) != "person":
            fail(f"{name}: expected a COCO head (91 category ids), got {len(labels)} labels")
    # With a static input the backbone's position embeddings are a constant: compute them once in
    # eager mode (exact) instead of tracing the antialiased bicubic resize, which the ONNX exporter
    # cannot express.
    frozen = 0
    for mod in model.modules():
        if hasattr(mod, "interpolate_pos_encoding") and hasattr(mod, "position_embeddings") and hasattr(mod, "patch_size"):
            n = (imgsz // mod.patch_size) ** 2
            with torch.no_grad():
                dummy = torch.zeros(1, n + 1, mod.position_embeddings.shape[-1])
                pe = mod.interpolate_pos_encoding(dummy, imgsz, imgsz).detach().clone()
            mod.interpolate_pos_encoding = types.MethodType(lambda self, e, h, w, pe=pe: pe, mod)
            frozen += 1
    if frozen == 0:
        fail(f"{name}: no position embeddings found to freeze (transformers changed?)")

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    wrapped = Wrapped(model).eval()
    import numpy as np

    x = synthetic_image(imgsz)
    x = ((x - np.array(IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]) / np.array(IMAGENET_STD, dtype=np.float32)[None, :, None, None]).astype(np.float32)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        logits, boxes = wrapped(xt)
    out: dict = {}
    onnx = args.work / f"{name}.onnx"
    log(f"{name}: exporting ONNX (opset {args.opset}, {imgsz}x{imgsz})")
    torch_export(wrapped, (xt,), ["pixel_values"], ["logits", "pred_boxes"], onnx, args.opset)
    if getattr(args, "release_model", False):
        # The training script after a long run: drop the PyTorch model (its reference outputs are
        # kept) before ONNX Runtime loads the file, so both never hold memory at once.
        import gc

        del wrapped, model, xt
        mod = None
        gc.collect()
    got = compare_with_onnxruntime(name, onnx, {"pixel_values": x}, {"logits": logits.numpy(), "pred_boxes": boxes.numpy()})
    pin_output_shapes(name, onnx, got)
    inputs, outputs = onnx_ports(onnx)
    log(f"{name}: " + check_ports(name, family, inputs, outputs, imgsz))
    top = lambda a: np.sort(1 / (1 + np.exp(-a.ravel())))[-20:]
    diff = float(np.abs(top(got["logits"]) - top(logits.numpy())).max())
    log(f"{name}: top-20 score difference PyTorch vs ONNX Runtime: {diff:.2e}")
    if diff > 1e-2:
        fail(f"{name}: ONNX Runtime and PyTorch disagree (top scores differ by {diff:.3f})")
    if names is not None and got["logits"].shape[-1] != nl:
        fail(f"{name}: logits have {got['logits'].shape[-1]} classes, expected {nl}")
    # A fine-tune's YAML lists its own classes (COCO spellings) in head order; the extra last logit
    # (rfdetr's num_classes + 1) has no name and is never reported as a known label.
    return finish_hf(name, onnx, args, out, names=names)


def load_yml(path: Path) -> dict:
    """A DEIMv2 training config with its `__include__` chain merged like the official YAMLConfig
    (included files first, the including file overrides; nested dicts merged)."""
    import yaml

    def merge(dst: dict, src: dict) -> dict:
        for k, v in src.items():
            if isinstance(v, dict) and isinstance(dst.get(k), dict):
                merge(dst[k], v)
            else:
                dst[k] = v
        return dst

    cfg = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    out: dict = {}
    for inc in cfg.pop("__include__", []):
        merge(out, load_yml((path.parent / inc).resolve()))
    return merge(out, cfg)


def deimv2_yml(name: str, src: Path, dinov3: bool) -> Path:
    """`configs/deimv2/deimv2_<hgnetv2|dinov3>_<size>_coco.yml` for `deimv2-<size>`."""
    size = name.lower().split("deimv2-", 1)[-1].split("_")[0]
    size = {"nano": "n", "small": "s", "medium": "m", "large": "l"}.get(size, size)
    path = src / "configs" / "deimv2" / f"deimv2_{'dinov3' if dinov3 else 'hgnetv2'}_{size}_coco.yml"
    if not path.is_file():
        fail(f"{name}: no training config {path} (name the model deimv2-<atto|femto|pico|n|s|m|l|x>)")
    return path


def deim_engine(name: str, src: Path, wanted: dict) -> dict:
    """Import the classes `wanted` (`class name -> module`) from the DEIMv2 source tree `src`,
    without running `engine/__init__.py` (data, optim, solver: training dependencies)."""
    eng = src.resolve() / "engine"
    if not (eng / "deim" / "postprocessor.py").is_file():
        fail(f"{name}: {src} is not a DEIMv2 source tree (engine/deim/postprocessor.py missing)")

    def package(mod_name: str, path: Path) -> types.ModuleType:
        m = sys.modules.get(mod_name)
        if m is None:
            m = types.ModuleType(mod_name)
            m.__path__ = [str(path)]
            sys.modules[mod_name] = m
        return m

    package("engine", eng)
    core = package("engine.core", eng / "core")
    package("engine.backbone", eng / "backbone")
    package("engine.deim", eng / "deim")
    core.register = importlib.import_module("engine.core.workspace").register
    return {cls: getattr(importlib.import_module(mod), cls) for cls, mod in wanted.items()}


def load_deimv2(name: str, src: Path, wdir: Path):
    """The DEIMv2 model of `wdir` (config.json + model.safetensors) built with the official code in
    `src`, importing only the modules the model needs (not the training stack). Module arguments
    are the official training config (`configs/deimv2/*.yml` and its includes), completed by the
    checkpoint's config.json: the Atto/Femto/Pico config.json files leave out settings the
    training config has (`activation: silu`, `mlp_act: silu`; the constructors default to relu),
    and the DINOv3-S/M ones have outdated `interaction_indexes`."""
    import inspect

    import torch.nn as nn

    classes = deim_engine(
        name,
        src,
        {
            "HGNetv2": "engine.backbone.hgnetv2",
            "DINOv3STAs": "engine.backbone.dinov3_adapter",
            "HybridEncoder": "engine.deim.hybrid_encoder",
            "LiteEncoder": "engine.deim.lite_encoder",
            "DEIMTransformer": "engine.deim.deim_decoder",
            "DFINETransformer": "engine.deim.dfine_decoder",
            "PostProcessor": "engine.deim.postprocessor",
        },
    )

    hf = json.loads((wdir / "config.json").read_text(encoding="utf-8"))
    hf = hf.get("config", hf)
    dinov3 = "DINOv3STAs" in hf
    yml_path = deimv2_yml(name, src, dinov3)
    yml = load_yml(yml_path)
    log(f"{name}: training config {yml_path.relative_to(src)}")
    deim = yml.get("DEIM", {})
    roles = {"backbone": deim.get("backbone"), "encoder": deim.get("encoder"), "decoder": deim.get("decoder")}
    for role, cls in roles.items():
        if cls not in hf:
            fail(f"{name}: the training config's {role} {cls} is not in config.json ({sorted(hf)})")

    def build(key: str, **force):
        cls = classes[key]
        params = inspect.signature(cls.__init__).parameters
        args = {k: v for k, v in yml.get(key, {}).items() if k in params}
        # Values the training framework shares into modules (`__share__`).
        for k in getattr(cls, "__share__", []):
            if k in yml and k in params:
                args.setdefault(k, yml[k])
        for k, v in hf.get(key, {}).items():
            if k in args and args[k] != v:
                if k in force:
                    continue
                # The training config wins: the published config.json files predate its fixes
                # (DINOv3-S/M interaction_indexes [3, 7, 11], "fix config error", 2025-10-12;
                # checked on COCO images: recall 0.48 / 0.52 vs 0.45 / 0.46 with config.json's).
                log(f"{name}: {key}.{k}: training config {args[k]!r} (config.json has {v!r})")
            else:
                args[k] = v
        args.update(force)
        return cls(**args)

    class DEIMv2(nn.Module):
        """The layout of the official Hugging Face checkpoints (DEIMv2's hf_models.ipynb)."""

        def __init__(self):
            super().__init__()
            if dinov3:
                # No pretrained-backbone file: everything comes from the safetensors.
                self.backbone = build("DINOv3STAs", weights_path=None)
            else:
                self.backbone = build("HGNetv2", pretrained=False)
            self.encoder = build(roles["encoder"])
            self.decoder = build(roles["decoder"])
            # Deploy mode returns COCO-80 indices before any category remapping.
            self.postprocessor = build("PostProcessor", remap_mscoco_category=False)

    model = DEIMv2()
    cfg = hf
    from safetensors.torch import load_model

    # load_model knows the tied tensors safetensors stored once (shared heads, decoder constants).
    missing, unexpected = load_model(model, str(wdir / "model.safetensors"), strict=False)
    if unexpected:
        fail(f"{name}: unexpected tensors in the checkpoint: {unexpected[:5]}")
    state = model.state_dict(keep_vars=True)
    real_missing = [k for k in missing if not any(state[k] is v for o, v in state.items() if o != k and o not in missing)]
    if real_missing:
        fail(f"{name}: tensors missing from the checkpoint: {real_missing[:5]}")
    size = getattr(model.decoder, "eval_spatial_size", None)
    if not size or size[0] != size[1]:
        fail(f"{name}: config.json has no square eval_spatial_size ({size})")
    pp = model.postprocessor
    if pp.num_classes != len(COCO80) or getattr(pp, "remap_mscoco_category", False):
        fail(f"{name}: expected 80 COCO-80 classes without category remapping")
    return model, int(size[0]), dinov3


def export_deimv2(name: str, family: str, imgsz: int | None, args: argparse.Namespace) -> dict:
    import numpy as np
    import torch

    if args.source is None:
        fail(f"{name}: --source <DEIMv2 source checkout> is required")
    wdir = weights_dir(name, args)
    log(f"{name}: loading {wdir} with the DEIMv2 code in {args.source}")
    model, size, dinov3 = load_deimv2(name, args.source, wdir)
    if imgsz is not None and imgsz != size:
        fail(f"{name}: --imgsz {imgsz} but the model was trained at {size} (eval_spatial_size)")
    imgsz = size
    model.eval()
    # Deploy mode (tools/deployment/export_onnx.py): reparameterized modules, tuple outputs.
    for m in model.modules():
        if hasattr(m, "convert_to_deploy"):
            m.convert_to_deploy()
    model.postprocessor.deploy()

    class Deploy(torch.nn.Module):
        """`images` (RGB 0..1) + `orig_target_sizes` -> labels, boxes (xyxy in target-size pixels),
        scores. The DINOv3 variants were trained on ImageNet-normalized input: done here."""

        def __init__(self, m, normalize: bool):
            super().__init__()
            self.m = m
            self.normalize = normalize
            self.register_buffer("mean", torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1))
            self.register_buffer("std", torch.tensor(IMAGENET_STD).view(1, 3, 1, 1))

        def forward(self, images, orig_target_sizes):
            x = (images - self.mean) / self.std if self.normalize else images
            outputs = self.m.decoder(self.m.encoder(self.m.backbone(x)))
            return self.m.postprocessor(outputs, orig_target_sizes)

    deploy = Deploy(model, dinov3).eval()
    log(f"{name}: {imgsz}x{imgsz}, {'DINOv3 (ImageNet normalization in the graph)' if dinov3 else 'HGNetv2'}")
    x = synthetic_image(imgsz)
    sizes = np.array([[imgsz, imgsz]], dtype=np.int64)
    with torch.no_grad():
        labels, boxes, scores = deploy(torch.from_numpy(x), torch.from_numpy(sizes))
    onnx = args.work / f"{name}.onnx"
    log(f"{name}: exporting ONNX (opset {args.opset})")
    torch_export(
        deploy,
        (torch.from_numpy(x), torch.from_numpy(sizes)),
        ["images", "orig_target_sizes"],
        ["labels", "boxes", "scores"],
        onnx,
        args.opset,
    )
    got = compare_with_onnxruntime(
        name,
        onnx,
        {"images": x, "orig_target_sizes": sizes},
        {"labels": labels.numpy(), "boxes": boxes.numpy(), "scores": scores.numpy()},
    )
    pin_output_shapes(name, onnx, got)
    inputs, outputs = onnx_ports(onnx)
    log(f"{name}: " + check_ports(name, family, inputs, outputs, imgsz))
    if got["labels"].min() < 0 or got["labels"].max() >= len(COCO80):
        fail(f"{name}: labels outside 0..79")
    diff = float(np.abs(np.sort(got["scores"].ravel())[-20:] - np.sort(scores.numpy().ravel())[-20:]).max())
    log(f"{name}: top-20 score difference PyTorch vs ONNX Runtime: {diff:.2e}")
    if diff > 1e-2:
        fail(f"{name}: ONNX Runtime and PyTorch disagree (top scores differ by {diff:.3f})")
    return finish_hf(name, onnx, args, {}, imgsz)


def rtdetrv2_checkpoint(name: str, args: argparse.Namespace) -> Path:
    """The `.pth` of an `rtdetrv2` export: --weights itself, or the one `.pth` in that directory."""
    if args.weights is None:
        fail(f"{name}: --weights <the .pth checkpoint or its directory> is required (nothing is downloaded)")
    w = args.weights.resolve()
    if w.is_dir():
        found = sorted(w.glob("*.pth"))
        if len(found) != 1:
            fail(f"{name}: expected exactly one .pth in {w}, found {len(found)}")
        w = found[0]
    if not w.is_file():
        fail(f"{name}: --weights {w} does not exist")
    return w


def load_rtdetrv2(name: str, src: Path, ckpt: Path, spec: dict):
    """An RT-DETRv2 checkpoint built with the DEIMv2 code's RT-DETRv2 modules (copied from
    lyuwenyu/RT-DETR): the training config `spec["config"]` with `num_classes` from the class list,
    loaded strictly from the checkpoint's EMA weights (else `model`). The checkpoint is read with
    `torch.load(weights_only=True)` (tensors and plain containers only, no pickled code)."""
    import inspect
    from collections import OrderedDict

    import torch
    import torch.nn as nn

    classes = deim_engine(
        name,
        src,
        {
            "PResNet": "engine.backbone.presnet",
            "HybridEncoder": "engine.deim.hybrid_encoder",
            "RTDETRTransformerv2": "engine.deim.rtdetrv2_decoder",
            "PostProcessor": "engine.deim.postprocessor",
        },
    )
    yml_path = src / spec["config"]
    if not yml_path.is_file():
        fail(f"{name}: no training config {yml_path}")
    yml = load_yml(yml_path)
    log(f"{name}: training config {yml_path.relative_to(src)}")
    # The checkpoint's dataset settings (CameraTraps megadetector_detection.yml).
    yml["num_classes"] = len(spec["names"])
    yml["remap_mscoco_category"] = False
    yml["eval_spatial_size"] = [spec["imgsz"], spec["imgsz"]]
    deim = yml.get("DEIM", {})
    roles = [deim.get("backbone"), deim.get("encoder"), deim.get("decoder")]
    if roles != ["PResNet", "HybridEncoder", "RTDETRTransformerv2"]:
        fail(f"{name}: {yml_path.name} is not an RT-DETRv2 config ({roles})")

    def build(key: str, **force):
        cls = classes[key]
        params = inspect.signature(cls.__init__).parameters
        kw = {k: v for k, v in yml.get(key, {}).items() if k in params}
        for k in getattr(cls, "__share__", []):
            if k in yml and k in params:
                kw.setdefault(k, yml[k])
        kw.update(force)
        return cls(**kw)

    class RTDETRv2(nn.Module):
        """RT-DETR's `RTDETR` module layout: backbone, encoder, decoder (checkpoint key prefixes)."""

        def __init__(self):
            super().__init__()
            self.backbone = build("PResNet", pretrained=False)
            self.encoder = build("HybridEncoder")
            self.decoder = build("RTDETRTransformerv2")

    model = RTDETRv2()
    # RT-DETRv2 projects every level with a 1x1 conv + BN, also when its width already equals
    # hidden_dim (the r18vd 256-channel level); DEIM's copy uses an identity there.
    hidden = model.encoder.hidden_dim
    for i, proj in enumerate(model.encoder.input_proj):
        if isinstance(proj, nn.Identity):
            model.encoder.input_proj[i] = nn.Sequential(
                OrderedDict(
                    [
                        ("conv", nn.Conv2d(model.encoder.in_channels[i], hidden, kernel_size=1, bias=False)),
                        ("norm", nn.BatchNorm2d(hidden)),
                    ]
                )
            )
    post = build("PostProcessor")
    ck = torch.load(str(ckpt), map_location="cpu", weights_only=True)
    if not isinstance(ck, dict):
        fail(f"{name}: {ckpt.name} is not an RT-DETR training checkpoint")
    if isinstance(ck.get("ema"), dict) and "module" in ck["ema"]:
        state, which = ck["ema"]["module"], "EMA"
    elif "model" in ck:
        state, which = ck["model"], "model"
    else:
        fail(f"{name}: {ckpt.name} has neither `ema` nor `model` weights ({sorted(ck)[:8]})")
    head = state.get("decoder.enc_score_head.weight")
    if head is None or head.shape[0] != len(spec["names"]):
        fail(f"{name}: the checkpoint's head has {None if head is None else head.shape[0]} classes, expected {len(spec['names'])}")
    model.load_state_dict(state, strict=True)
    log(f"{name}: loaded the {which} weights of {ckpt.name} ({len(state)} tensors, strict)")
    if post.num_classes != len(spec["names"]) or post.remap_mscoco_category:
        fail(f"{name}: post-processor has {post.num_classes} classes (remap {post.remap_mscoco_category})")
    return model, post


def export_rtdetrv2(name: str, family: str, imgsz: int | None, args: argparse.Namespace) -> dict:
    import numpy as np
    import torch

    spec = RTDETRV2_MODELS.get(name.lower())
    if spec is None:
        fail(f"{name}: not an RT-DETRv2 checkpoint this script knows ({sorted(RTDETRV2_MODELS)})")
    if args.source is None:
        fail(f"{name}: --source <DEIMv2 source checkout> is required (its RT-DETRv2 modules build the model)")
    if imgsz is not None and imgsz != spec["imgsz"]:
        fail(f"{name}: --imgsz {imgsz} but the model was trained at {spec['imgsz']}")
    imgsz = spec["imgsz"]
    ckpt = rtdetrv2_checkpoint(name, args)
    log(f"{name}: loading {ckpt} with the RT-DETRv2 modules of the DEIMv2 code in {args.source}")
    model, post = load_rtdetrv2(name, args.source, ckpt, spec)
    model.eval()
    for m in model.modules():
        if hasattr(m, "convert_to_deploy"):
            m.convert_to_deploy()
    post.deploy()

    class Deploy(torch.nn.Module):
        """`images` (RGB 0..1, stretched to S x S like PytorchWildlife's `Resize`) +
        `orig_target_sizes` ([w, h]) -> labels, boxes (xyxy in target-size pixels), scores."""

        def __init__(self, m, p):
            super().__init__()
            self.m = m
            self.p = p

        def forward(self, images, orig_target_sizes):
            return self.p(self.m.decoder(self.m.encoder(self.m.backbone(images))), orig_target_sizes)

    deploy = Deploy(model, post).eval()
    x = synthetic_image(imgsz)
    sizes = np.array([[imgsz, imgsz]], dtype=np.int64)
    with torch.no_grad():
        labels, boxes, scores = deploy(torch.from_numpy(x), torch.from_numpy(sizes))
    onnx = args.work / f"{name}.onnx"
    log(f"{name}: exporting ONNX (opset {args.opset}, {imgsz}x{imgsz})")
    torch_export(
        deploy,
        (torch.from_numpy(x), torch.from_numpy(sizes)),
        ["images", "orig_target_sizes"],
        ["labels", "boxes", "scores"],
        onnx,
        args.opset,
    )
    got = compare_with_onnxruntime(
        name,
        onnx,
        {"images": x, "orig_target_sizes": sizes},
        {"labels": labels.numpy(), "boxes": boxes.numpy(), "scores": scores.numpy()},
    )
    pin_output_shapes(name, onnx, got)
    inputs, outputs = onnx_ports(onnx)
    log(f"{name}: " + check_ports(name, family, inputs, outputs, imgsz))
    nc = len(spec["names"])
    if got["labels"].min() < 0 or got["labels"].max() >= nc:
        fail(f"{name}: labels outside 0..{nc - 1}")
    diff = float(np.abs(np.sort(got["scores"].ravel())[-20:] - np.sort(scores.numpy().ravel())[-20:]).max())
    log(f"{name}: top-20 score difference PyTorch vs ONNX Runtime: {diff:.2e}")
    if diff > 1e-2:
        fail(f"{name}: ONNX Runtime and PyTorch disagree (top scores differ by {diff:.3f})")
    # The boxes of the 20 best PyTorch detections, each against the closest ONNX Runtime box of
    # the same label among its 50 best (near-equal scores may swap places in the top-k).
    ref_b, ref_l, ref_s = boxes.numpy()[0], labels.numpy()[0], scores.numpy()[0]
    got_b, got_l, got_s = got["boxes"][0], got["labels"][0], got["scores"][0]
    cand = np.argsort(got_s)[-50:]
    box_diff = 0.0
    for i in np.argsort(ref_s)[-20:]:
        same = [j for j in cand if got_l[j] == ref_l[i]]
        d = min((float(np.abs(got_b[j] - ref_b[i]).max()) for j in same), default=float("inf"))
        box_diff = max(box_diff, d)
    log(f"{name}: box difference of the 20 best detections: {box_diff:.2e} px")
    if box_diff > 0.5:
        fail(f"{name}: ONNX Runtime and PyTorch boxes disagree by {box_diff:.2f} px")
    return finish_hf(name, onnx, args, {}, imgsz, spec["names"])


def finish_hf(
    name: str, onnx: Path, args: argparse.Namespace, out: dict, imgsz: int | None = None, names: list[str] | None = None
) -> dict:
    if not args.no_onnx:
        out["onnx"] = args.out / f"{name}.onnx"
        shutil.copy2(onnx, out["onnx"])
        stamp_coreml_cache_key(out["onnx"])
    if not args.no_openvino:
        log(f"{name}: converting to OpenVINO IR")
        out["xml"] = onnx_to_ir(onnx, args.out, name, args.fp32)
    names = names or COCO80
    out["yaml"] = write_names(args.out, name, names, len(names))
    out["names"] = names
    if imgsz is not None:
        out["imgsz"] = imgsz
    return out


# Modules each recipe imports, in import order (`--self-check`).
SELF_CHECK_MODULES = {
    "ultralytics": ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "ultralytics"],
    "rfdetr": ["numpy", "torch", "onnx", "onnxruntime", "safetensors", "transformers"],
    "deimv2": ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "safetensors", "yaml"],
    "rtdetrv2": ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "yaml"],
}


def loaded_dll(name: str) -> str | None:
    """Path of the DLL `name` loaded into this process (Windows), else None."""
    if os.name != "nt":
        return None
    import ctypes
    from ctypes import wintypes

    k32 = ctypes.WinDLL("kernel32", use_last_error=True)
    k32.GetModuleHandleW.restype = wintypes.HMODULE
    k32.GetModuleHandleW.argtypes = [wintypes.LPCWSTR]
    k32.GetModuleFileNameW.argtypes = [wintypes.HMODULE, wintypes.LPWSTR, wintypes.DWORD]
    handle = k32.GetModuleHandleW(name)
    if not handle:
        return None
    buf = ctypes.create_unicode_buffer(1024)
    n = k32.GetModuleFileNameW(handle, buf, len(buf))
    return buf.value[:n] if n else None


def self_check(recipe: str) -> int:
    """Import what `recipe` needs, one module at a time, and print the versions: a broken package
    or a native library that cannot load fails here in seconds, naming the module (run with
    `-X faulthandler` so a native crash leaves a stack)."""
    log(f"self-check: Python {sys.version.split()[0]} ({sys.executable})")
    for mod in SELF_CHECK_MODULES[recipe]:
        log(f"self-check: importing {mod}")
        m = importlib.import_module(mod)
        log(f"self-check: {mod} {getattr(m, '__version__', '?')}")
    for dll in ("msvcp140.dll", "vcruntime140.dll", "vcruntime140_1.dll"):
        path = loaded_dll(dll)
        if path:
            log(f"self-check: {dll} loaded from {path}")
    log("self-check: ok")
    return 0


# ---------------------------------------------------------------------------------------------


def main() -> int:
    args = parse_args()
    if args.self_check:
        return self_check(args.recipe)
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    args.work = args.work.resolve()
    args.work.mkdir(parents=True, exist_ok=True)
    snippets = []
    for name in args.name:
        recipe = args.recipe or recipe_for(name)
        family = args.family or family_for(name, recipe)
        if recipe == "ultralytics":
            out = export_ultralytics(name, family, args.imgsz or 640, args)
        elif recipe == "rfdetr":
            out = export_rfdetr(name, family, args.imgsz or 704, args)
        elif recipe == "rtdetrv2":
            out = export_rtdetrv2(name, family, args.imgsz, args)
        else:
            out = export_deimv2(name, family, args.imgsz, args)
        # The .onnx runs on both runtimes (OpenVINO and ONNX Runtime, incl. CoreML); the IR only on OpenVINO.
        model_file = out.get("onnx") or out.get("xml")
        print(f"{name}: {recipe} -> {model_file} (family {family}, {len(out.get('names', COCO80))} classes)", flush=True)
        snippets.append(
            {"name": name, "path": f"models/{model_file.name}", "family": family, "classes": f"models/{name}.yaml"}
        )
    print(f"\nWrote to {args.out}. They are listed on the Config page under \"Local models (not in config)\"")
    print("when this is the server's models directory; or paste this into blue_onyx_prism_config.json -> \"models\":")
    print(json.dumps(snippets, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
