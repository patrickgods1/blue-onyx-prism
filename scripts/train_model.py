#!/usr/bin/env python
"""Fine-tune a detection model on a prepared dataset and export it to a static ONNX for Blue Onyx Prism.

Trainers (`--trainer`):

  ultralytics   Ultralytics YOLO26 / YOLO11 / YOLOv8 from an official `.pt` (AGPL-3.0, so is the
                result). `YOLO(<base>.pt).train(data=<data.yaml>, ...)`, then the model export's
                Ultralytics recipe (static, nms=False): YOLO26 -> `[1, 300, 6]` (family `yolo26`),
                YOLO11 / YOLOv8 -> `[1, 4 + C, A]` (family `yolo8`).
  rfdetr        RF-DETR nano / small / medium / base / large (Apache-2.0) with the `rfdetr` package:
                `RFDETRNano(pretrain_weights=<base .pth>).train(dataset_dir=<COCO dir>, ...)` on the
                COCO layout it expects (`train/`, `valid/`, `test/` with `_annotations.coco.json`,
                written here from the prepared dataset), then the model export's `rfdetr` recipe
                (transformers `RfDetrForObjectDetection` + torch.onnx, `pixel_values` -> `logits`,
                `pred_boxes`, family `rfdetr`), checked against rfdetr's own PyTorch model.
  transformers  RT-DETRv2 (`RTDetrV2ForObjectDetection`) and D-FINE (`DFineForObjectDetection`),
                Apache-2.0, with transformers' `Trainer` on the prepared COCO JSON, then torch.onnx:
                `pixel_values [1,3,S,S]` (RGB 0..1, no normalization) -> `logits [1,Q,C]`,
                `pred_boxes [1,Q,4]` (family `detr`).

Prepared dataset (`--data`, written by the service, see docs/TRAINING.md):

    classes.txt                 one class per line; class id = line number (0-based)
    images/<file>               every image, unique names
    labels/<stem>.txt           YOLO labels (`class cx cy w h`, normalized); empty = no objects
    train.txt, valid.txt        image paths relative to the dataset dir, one per line (train may
                                repeat a line: oversampling)
    data.yaml                   Ultralytics dataset file (path, train, val, names)
    train.coco.json, valid.coco.json   the same splits as COCO instances (category ids 0..C-1,
                                file_name relative to the dataset dir; repeated images are
                                separate image entries)

Progress: every event is one line `@train {json}` on stdout (`start`, `epoch`, `batch`, `phase`,
`done`); the service parses them, everything else is log output. Night augmentation
(`--aug-dark`, `--aug-noise`, `--aug-gray`: per-image probabilities) is applied on the fly to the
training batches (Ultralytics, transformers) or to one extra copy of each training image chosen by
those probabilities (rfdetr).

The service runs this file (embedded in the binary) inside its pinned training environment
(<data root>/tools/train-env), offline: YOLO_OFFLINE=1, HF_HUB_OFFLINE=1, TRANSFORMERS_OFFLINE=1;
the base weights were downloaded and SHA-256 checked by the service. `--export-script` is the
embedded scripts/export_model.py, whose functions do the export and the ONNX Runtime check.
By hand (inside an environment from scripts/requirements-model-train.txt):

    python scripts/train_model.py --trainer ultralytics --family yolo26 --size n --name my-yolo26n \\
        --weights yolo26n.pt --data <prepared dir> --imgsz 640 --epochs 50 --batch 16 \\
        --out-dir models --work work --export-script scripts/export_model.py
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import math
import os
import random
import shutil
import sys
import time
from pathlib import Path

TRAINERS = ("ultralytics", "rfdetr", "transformers")
FAMILIES = ("yolo26", "yolo11", "yolov8", "rfdetr", "rtdetrv2", "dfine")
RFDETR_CLASSES = {
    "nano": "RFDETRNano",
    "small": "RFDETRSmall",
    "medium": "RFDETRMedium",
    "base": "RFDETRBase",
    "large": "RFDETRLarge",
}


def log(msg: str) -> None:
    print(f"[train_model] {msg}", flush=True)


def event(kind: str, **fields) -> None:
    """One progress event for the service (`@train {json}`)."""
    fields["event"] = kind
    # A leading newline: progress bars (Ultralytics, Lightning) leave the cursor mid-line.
    print("\n@train " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"[train_model] error: {msg}")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--trainer", choices=TRAINERS, required=True)
    p.add_argument("--self-check", action="store_true", help="only import what --trainer needs, report the device and exit")
    p.add_argument("--family", choices=FAMILIES)
    p.add_argument("--size", help="n/s/m/l/x (YOLO, RT-DETRv2, D-FINE) or nano/small/medium/base/large (RF-DETR)")
    p.add_argument("--name", help="name of the trained model (<name>.onnx / <name>.yaml)")
    p.add_argument("--weights", type=Path, help="base weights: the .pt / .pth file, or the directory with model.safetensors + config.json")
    p.add_argument("--data", type=Path, help="prepared dataset directory")
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--epochs", type=int, default=50)
    p.add_argument("--batch", type=int, default=8)
    p.add_argument("--lr", type=float, default=None, help="learning rate (default: the trainer's)")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    p.add_argument("--allow-cpu", action="store_true", help="train on the CPU when no GPU is found (slow; smoke tests)")
    p.add_argument("--workers", type=int, default=2, help="data loader workers")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=0, help="stop after this many epochs without improvement (0 = off)")
    p.add_argument("--aug-dark", type=float, default=0.0, help="probability of the synthetic darkness augmentation")
    p.add_argument("--aug-noise", type=float, default=0.0, help="probability of the sensor noise augmentation")
    p.add_argument("--aug-gray", type=float, default=0.0, help="probability of the grayscale (IR-like) augmentation")
    p.add_argument("--out-dir", type=Path, help="where <name>.onnx and <name>.yaml are written")
    p.add_argument("--work", type=Path, help="scratch directory (checkpoints, logs)")
    p.add_argument("--export-script", type=Path, help="scripts/export_model.py (its functions do the export)")
    p.add_argument("--opset", type=int, default=17)
    a = p.parse_args()
    if a.self_check:
        return a
    missing = [k for k in ("family", "size", "name", "weights", "data", "out_dir", "work", "export_script") if getattr(a, k) is None]
    if missing:
        p.error("missing: " + ", ".join("--" + m.replace("_", "-") for m in missing))
    for k in ("aug_dark", "aug_noise", "aug_gray"):
        v = getattr(a, k)
        if not 0.0 <= v <= 1.0:
            p.error(f"--{k.replace('_', '-')} must be between 0 and 1")
    # The input size step is the family's (32; RF-DETR base 56): checked by the trainers.
    if a.epochs < 1 or a.batch < 1 or a.imgsz < 32:
        p.error("--epochs and --batch must be at least 1, --imgsz at least 32")
    return a


SELF_CHECK_MODULES = {
    "ultralytics": ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "ultralytics"],
    "rfdetr": ["numpy", "torch", "torchvision", "onnx", "onnxruntime", "transformers", "pytorch_lightning", "rfdetr"],
    "transformers": ["numpy", "torch", "onnx", "onnxruntime", "safetensors", "accelerate", "transformers"],
}


def device_info() -> dict:
    import torch

    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(), "mps": False}
    if info["cuda"]:
        info["cuda_version"] = torch.version.cuda
        info["devices"] = [
            {"name": torch.cuda.get_device_name(i), "capability": list(torch.cuda.get_device_capability(i)),
             "vram_mb": torch.cuda.get_device_properties(i).total_memory // (1024 * 1024)}
            for i in range(torch.cuda.device_count())
        ]
        info["arch_list"] = torch.cuda.get_arch_list()
    mps = getattr(torch.backends, "mps", None)
    info["mps"] = bool(mps is not None and mps.is_available())
    return info


def self_check(trainer: str, device: str = "auto") -> int:
    log(f"self-check: Python {sys.version.split()[0]} ({sys.executable})")
    for mod in SELF_CHECK_MODULES[trainer]:
        log(f"self-check: importing {mod}")
        m = importlib.import_module(mod)
        log(f"self-check: {mod} {getattr(m, '__version__', '?')}")
    info = device_info()
    if info["cuda"]:
        # A GPU the wheel has no kernels for fails at the first kernel launch: try one now.
        import torch

        # On the run's GPU (--device cuda:N): another GPU of the machine may have other kernels.
        want = device.strip().lower()
        dev = want if want.startswith("cuda:") else "cuda"
        if dev.startswith("cuda:") and int(dev[5:]) >= torch.cuda.device_count():
            fail(f"--device {device}: PyTorch sees {torch.cuda.device_count()} CUDA device(s); is CUDA_VISIBLE_DEVICES different from the service's?")
        x = torch.ones(8, device=dev)
        info["cuda_ok"] = float((x * 2).sum().item()) == 16.0
        info["device"] = dev
    event("self_check", trainer=trainer, **info)
    log("self-check: ok")
    return 0


def pick_device(want: str, allow_cpu: bool) -> str:
    import torch

    w = want.strip().lower()
    if w == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        if allow_cpu:
            return "cpu"
        fail("no CUDA or Apple MPS device is available to PyTorch; training needs a GPU (pass --allow-cpu for a slow smoke test)")
    if w == "cuda":
        w = "cuda:0"
    if w.startswith("cuda") and not torch.cuda.is_available():
        fail(f"--device {want}: PyTorch sees no CUDA device (driver too old for CUDA 12, or no NVIDIA GPU)")
    if w == "mps" and not torch.backends.mps.is_available():
        fail("--device mps: MPS is not available (macOS 14+ on Apple silicon)")
    if w == "cpu" and not allow_cpu:
        fail("--device cpu needs --allow-cpu")
    return w


def watch_parent() -> None:
    """Exit when the process that started this one goes away (the service stopped or crashed):
    on Unix this process leads its own process group, which the service kills on cancel and stop,
    but not when the service itself is killed. Windows: the service's job object does it."""
    if os.name == "nt":
        return
    ppid = os.getppid()
    if ppid <= 1:
        return
    import threading

    def loop() -> None:
        while True:
            time.sleep(2)
            if os.getppid() != ppid:
                log("the parent process is gone; stopping")
                os._exit(3)

    threading.Thread(target=loop, name="parent-watch", daemon=True).start()


def slice_logits(name: str, path: Path, k: int) -> None:
    """Cut an RF-DETR graph's `logits [1, Q, K + 1]` to the K named classes (rfdetr adds one
    unnamed logit; the service would report it as `class_K`)."""
    import numpy as np
    import onnx
    from onnx import helper, numpy_helper

    model = onnx.load(str(path))
    g = model.graph
    out = next((o for o in g.output if o.name == "logits"), None)
    if out is None:
        fail(f"{name}: no `logits` output to slice")
    dims = out.type.tensor_type.shape.dim
    if len(dims) != 3 or dims[2].dim_value != k + 1:
        fail(f"{name}: logits are {[d.dim_value for d in dims]}, expected [1, Q, {k + 1}]")
    for node in g.node:
        node.output[:] = ["logits_full" if o == "logits" else o for o in node.output]
    for nm, v in (("logits_slice_starts", 0), ("logits_slice_ends", k), ("logits_slice_axes", 2)):
        g.initializer.append(numpy_helper.from_array(np.array([v], dtype=np.int64), nm))
    g.node.append(
        helper.make_node(
            "Slice", ["logits_full", "logits_slice_starts", "logits_slice_ends", "logits_slice_axes"], ["logits"], name="logits_named_classes"
        )
    )
    dims[2].dim_value = k
    onnx.checker.check_model(model)
    onnx.save(model, str(path))
    log(f"{name}: logits cut to the {k} named classes")


def load_export_module(path: Path):
    """scripts/export_model.py as a module (the interpreter runs with -I, so it is not on sys.path)."""
    spec = importlib.util.spec_from_file_location("export_model", str(path))
    if spec is None or spec.loader is None:
        fail(f"cannot load {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_classes(data: Path) -> list[str]:
    names = [l.strip() for l in (data / "classes.txt").read_text(encoding="utf-8").splitlines() if l.strip()]
    if not names:
        fail(f"{data / 'classes.txt'} lists no classes")
    return names


def read_split(data: Path, split: str) -> list[str]:
    p = data / f"{split}.txt"
    if not p.is_file():
        fail(f"{p} is missing")
    return [l.strip() for l in p.read_text(encoding="utf-8").splitlines() if l.strip()]


# ---------------------------------------------------------------------------------------------
# Night augmentation (synthetic darkness, sensor noise, grayscale)
# ---------------------------------------------------------------------------------------------


def night_augment(x, p_dark: float, p_noise: float, p_gray: float, gen=None):
    """Night augmentation of RGB float images in 0..1, `[3,H,W]` or `[B,3,H,W]` (torch, any device).

    Per image, independently: grayscale (BT.601 luma in all channels, like a camera in IR mode)
    with `p_gray`; darkness with `p_dark` (to linear light with gamma 2.2, exposure x0.05..0.4,
    back: an under-exposed night frame, MAET-style); then sensor noise with `p_noise` (shot noise
    proportional to the signal plus read noise, sigma 0.01..0.05). Returns a new tensor."""
    import torch

    single = x.dim() == 3
    if single:
        x = x.unsqueeze(0)
    b = x.shape[0]
    dev = x.device

    def coin(p):
        return (torch.rand(b, 1, 1, 1, generator=gen) < p).to(dev)

    def unif(lo, hi):
        return (lo + (hi - lo) * torch.rand(b, 1, 1, 1, generator=gen)).to(dev)

    y = x.clamp(0, 1)
    if p_gray > 0:
        luma = (0.299 * y[:, 0:1] + 0.587 * y[:, 1:2] + 0.114 * y[:, 2:3]).expand_as(y)
        y = torch.where(coin(p_gray), luma, y)
    if p_dark > 0:
        dark = (y.pow(2.2) * unif(0.05, 0.4)).pow(1 / 2.2)
        y = torch.where(coin(p_dark), dark, y)
    if p_noise > 0:
        sigma = unif(0.01, 0.05)
        noise = torch.randn(y.shape, generator=gen).to(dev)
        noisy = y + noise * (sigma * y.clamp(min=1e-4).sqrt() * 0.5 + sigma * 0.5)
        y = torch.where(coin(p_noise), noisy, y)
    y = y.clamp(0, 1)
    return y[0] if single else y


def any_augment(a) -> bool:
    return a.aug_dark > 0 or a.aug_noise > 0 or a.aug_gray > 0


# ---------------------------------------------------------------------------------------------
# Class head initialization from the COCO head
# ---------------------------------------------------------------------------------------------

# Other spellings of COCO class names (VOC, ExDark, common usage).
COCO_ALIASES = {
    "motorbike": "motorcycle",
    "people": "person",
    "pedestrian": "person",
    "bike": "bicycle",
    "aeroplane": "airplane",
    "sofa": "couch",
    "tvmonitor": "tv",
    "diningtable": "dining table",
    "table": "dining table",
    "pottedplant": "potted plant",
}
# COCO category ids (the 91-wide RF-DETR head) of the 80 COCO classes, in COCO-80 order.
COCO80_CATEGORY_IDS = [
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31,
    32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57,
    58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87,
    88, 89, 90,
]


def head_rows(names: list[str], base_names: list[str]) -> list:
    """For each class of the run, the row of the base model's class head with the same class
    (case-insensitive, COCO aliases), or None (a new class, e.g. raccoon)."""
    base = [str(n).strip().lower() for n in base_names]
    rows = []
    for n in names:
        key = str(n).strip().lower()
        key = COCO_ALIASES.get(key, key)
        rows.append(base.index(key) if key in base else None)
    return rows


def copy_head_rows(new_sd: dict, base_sd: dict, rows: list, base_n: int) -> int:
    """Start the new class head from the base head's rows of the same classes: every tensor of
    `new_sd` whose first dimension is the run's class count where the base's is `base_n` (or both
    one more: a padding / background row, copied too) gets the mapped rows. Rows of new classes
    keep their fresh init. In place; returns the number of tensors changed."""
    import torch

    k = len(rows)
    changed = 0
    with torch.no_grad():
        for name, t in new_sd.items():
            b = base_sd.get(name)
            if b is None or not hasattr(b, "shape") or b.dim() == 0 or b.shape == t.shape or b.shape[1:] != t.shape[1:]:
                continue
            if b.shape[0] == base_n and t.shape[0] == k:
                extra = False
            elif b.shape[0] == base_n + 1 and t.shape[0] == k + 1:
                extra = True
            else:
                continue
            b = b.to(device=t.device, dtype=t.dtype)
            for i, j in enumerate(rows):
                if j is not None:
                    t[i].copy_(b[j])
            if extra:
                t[k].copy_(b[base_n])
            changed += 1
    return changed


def report_head(names: list[str], rows: list, changed: int) -> None:
    mapped = [n for n, r in zip(names, rows) if r is not None]
    new = [n for n, r in zip(names, rows) if r is None]
    log(f"class head: {changed} tensor(s) started from the base model's rows for {mapped or 'no class'}; new: {new or 'none'}")
    event("head_init", mapped=mapped, new=new, tensors=changed)


# ---------------------------------------------------------------------------------------------
# Ultralytics
# ---------------------------------------------------------------------------------------------


def train_ultralytics(a, device: str, names: list[str], em) -> dict:
    import torch
    from ultralytics import YOLO
    from ultralytics.models.yolo.detect import DetectionTrainer
    import ultralytics.engine.trainer as engine_trainer

    # The AMP check downloads yolo26n.pt (offline: it fails and warns); decide here instead, with
    # the same list of GPUs Ultralytics knows to produce NaN losses with AMP.
    def check_amp(model):
        dev = next(model.parameters()).device
        if dev.type != "cuda":
            return False
        bad = ("1660", "1650", "1630", "t400", "t550", "t600", "t1000", "t1200", "t2000", "k40m")
        name = torch.cuda.get_device_name(dev).lower()
        return not any(b in name for b in bad)

    engine_trainer.check_amp = check_amp

    def coco_sized_model(trainer, cfg, weights, k):
        import torch.nn as nn
        from ultralytics.nn.tasks import DetectionModel

        model = DetectionModel(cfg, nc=len(base_names), ch=trainer.data["channels"], verbose=False)
        model.names = dict(enumerate(base_names))
        model.load(weights)
        det = model.model[-1]
        rows = head_rows(names, base_names)
        swapped = 0
        for attr in ("cv3", "one2one_cv3"):
            for seq in getattr(det, attr, None) or []:
                old = seq[-1]
                new = nn.Conv2d(old.in_channels, k, 1).to(device=old.weight.device, dtype=old.weight.dtype)
                with torch.no_grad():
                    # New classes: the mean prior of the COCO classes (about 1% confidence).
                    new.bias.fill_(float(old.bias.mean()))
                    for i, j in enumerate(rows):
                        if j is not None:
                            new.weight[i].copy_(old.weight[j])
                            new.bias[i] = old.bias[j]
                seq[-1] = new
                swapped += 1
        if swapped == 0:
            raise RuntimeError("no class conv found in the detection head")
        det.nc = k
        det.no = k + det.reg_max * 4
        model.nc = k
        model.yaml["nc"] = k
        model.names = trainer.data["names"]
        surgery["done"] = True
        report_head(names, rows, swapped)
        return model
    probs = (a.aug_dark, a.aug_noise, a.aug_gray)
    gen = torch.Generator().manual_seed(a.seed)

    base_yolo = YOLO(str(a.weights))
    base_names = em.ultralytics_names(base_yolo)
    surgery = {"done": False}

    class NightTrainer(DetectionTrainer):
        def get_model(self, cfg=None, weights=None, verbose=True):
            # Ultralytics sizes the class branch's hidden width by the class count
            # (max(ch[0], min(nc, 100))), so a 3-class model shares none of it with the COCO model
            # and starts the whole branch from scratch. Instead: build the COCO-sized model, load
            # every tensor, then replace only the last class conv by one with a row per class,
            # started from the COCO row of the same class (fresh for new classes).
            k = len(names)
            if weights is None or k == len(base_names):
                return super().get_model(cfg, weights, verbose)
            try:
                return coco_sized_model(self, cfg, weights, k)
            except Exception as e:  # any surprise in Ultralytics' internals: its own path
                log(f"class head: keeping Ultralytics' fresh head ({type(e).__name__}: {e})")
                return super().get_model(cfg, weights, verbose)

        def preprocess_batch(self, batch):
            batch = super().preprocess_batch(batch)
            if any(p > 0 for p in probs):
                batch["img"] = night_augment(batch["img"], *probs, gen=gen)
            return batch

    os.chdir(a.work)
    weights = a.work / f"base-{a.weights.name}"
    shutil.copyfile(a.weights, weights)
    model = YOLO(str(weights))
    started = time.time()
    state = {"batches": 0}

    def on_train_epoch_start(trainer):
        state["batches"] = 0
        state["epoch_started"] = time.time()

    def on_train_batch_end(trainer):
        state["batches"] += 1
        nb = len(trainer.train_loader)
        if state["batches"] == nb or state["batches"] % max(1, nb // 10) == 0:
            event("batch", epoch=trainer.epoch + 1, epochs=trainer.epochs, batch=state["batches"], batches=nb)

    def on_fit_epoch_end(trainer):
        # Also called once more by the final validation of best.pt: report each epoch once.
        if trainer.epoch + 1 <= state.get("reported", 0) or trainer.epoch + 1 > trainer.epochs:
            return
        state["reported"] = trainer.epoch + 1
        m = {k: float(v) for k, v in (trainer.metrics or {}).items() if isinstance(v, (int, float))}
        tl = getattr(trainer, "tloss", None)
        if isinstance(tl, dict):
            loss = float(sum(float(v) for v in tl.values()))
        elif tl is not None:
            loss = float(tl.sum())
        else:
            loss = None
        event(
            "epoch",
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            loss=loss,
            map50=m.get("metrics/mAP50(B)"),
            map=m.get("metrics/mAP50-95(B)"),
            elapsed_s=round(time.time() - started, 1),
        )

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_fit_epoch_end", on_fit_epoch_end)

    base_model = base_yolo.model

    def on_pretrain_routine_end(trainer):
        # Without the head swap (see NightTrainer.get_model), start the known classes' last conv
        # rows from the COCO head where the shapes allow (Ultralytics also remaps biases by name).
        if surgery["done"]:
            return
        rows = head_rows(names, base_names)
        if len(base_names) == len(names) and rows == list(range(len(names))):
            return
        base_sd = base_model.float().state_dict()
        changed = copy_head_rows(trainer.model.state_dict(), base_sd, rows, len(base_names))
        ema = getattr(trainer, "ema", None)
        if ema is not None and getattr(ema, "ema", None) is not None:
            copy_head_rows(ema.ema.state_dict(), base_sd, rows, len(base_names))
        report_head(names, rows, changed)

    model.add_callback("on_pretrain_routine_end", on_pretrain_routine_end)
    kwargs = dict(
        data=str(a.data / "data.yaml"),
        epochs=a.epochs,
        imgsz=a.imgsz,
        batch=a.batch,
        device=device if device != "cuda:0" else 0,
        workers=a.workers,
        project=str(a.work),
        name="run",
        exist_ok=True,
        plots=False,
        seed=a.seed,
        deterministic=False,
        patience=a.patience or 100,
        trainer=NightTrainer,
        verbose=True,
    )
    if a.lr is not None:
        kwargs["lr0"] = a.lr
        kwargs["optimizer"] = "AdamW"
    model.train(**kwargs)
    best = a.work / "run" / "weights" / "best.pt"
    if not best.is_file():
        best = a.work / "run" / "weights" / "last.pt"
    if not best.is_file():
        fail("Ultralytics finished without writing best.pt or last.pt")
    event("phase", phase="export", detail=f"exporting {best.name} to ONNX")
    service_family = "yolo26" if a.family == "yolo26" else "yolo8"
    ex = argparse.Namespace(
        weights=best, work=a.work / "export", out=a.out_dir, opset=a.opset, no_onnx=False, no_openvino=True,
        int8=False, fp32=False, data="",
    )
    ex.work.mkdir(parents=True, exist_ok=True)
    out = em.export_ultralytics(a.name, service_family, a.imgsz, ex)
    if out["names"] != names:
        fail(f"the trained model has classes {out['names']}, the dataset {names}")
    return {"family": service_family, **out}


# ---------------------------------------------------------------------------------------------
# RF-DETR (rfdetr package)
# ---------------------------------------------------------------------------------------------


def link_or_copy(src: Path, dst: Path) -> None:
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except OSError:
        shutil.copyfile(src, dst)


def rfdetr_layout(a, names: list[str]) -> Path:
    """The COCO layout the rfdetr package reads (`train/`, `valid/`, `test/` with the images and
    `_annotations.coco.json`), from the prepared COCO JSON. `test` is the validation split
    (the held-out images are benchmarked by the service, never shown to the trainer). With night
    augmentation, the training split gets one augmented copy of each image the probabilities pick."""
    import numpy as np
    import torch
    from PIL import Image

    root = a.work / "rfdetr-data"
    gen = torch.Generator().manual_seed(a.seed)
    for split, src_name in (("train", "train"), ("valid", "valid"), ("test", "valid")):
        coco = json.loads((a.data / f"{src_name}.coco.json").read_text(encoding="utf-8"))
        cats = [c["name"] for c in sorted(coco["categories"], key=lambda c: c["id"])]
        if cats != names or [c["id"] for c in sorted(coco["categories"], key=lambda c: c["id"])] != list(range(len(names))):
            fail(f"{src_name}.coco.json categories {cats} do not match classes.txt {names} (ids 0..{len(names) - 1})")
        d = root / split
        d.mkdir(parents=True, exist_ok=True)
        images, anns = [], list(coco["annotations"])
        by_image: dict[int, list] = {}
        for ann in coco["annotations"]:
            by_image.setdefault(ann["image_id"], []).append(ann)
        next_id = max([i["id"] for i in coco["images"]] + [0]) + 1
        next_ann = max([x["id"] for x in anns] + [0]) + 1
        for im in coco["images"]:
            src = a.data / im["file_name"]
            # One file per source image; repeated entries (oversampling) share it.
            flat = Path(im["file_name"]).name
            link_or_copy(src, d / flat)
            images.append({**im, "file_name": flat})
            if split == "train" and any_augment(a):
                with Image.open(src) as pil:
                    x = torch.from_numpy(np.asarray(pil.convert("RGB"), dtype=np.float32) / 255).permute(2, 0, 1)
                y = night_augment(x, a.aug_dark, a.aug_noise, a.aug_gray, gen=gen)
                if not torch.equal(x, y):
                    aug = f"aug{next_id}_{flat}"
                    out = (y.permute(1, 2, 0).numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8)
                    Image.fromarray(out).save(d / aug, quality=92)
                    images.append({**im, "id": next_id, "file_name": aug})
                    for ann in by_image.get(im["id"], []):
                        anns.append({**ann, "id": next_ann, "image_id": next_id})
                        next_ann += 1
                    next_id += 1
        out = {"images": images, "annotations": anns, "categories": coco["categories"]}
        (d / "_annotations.coco.json").write_text(json.dumps(out), encoding="utf-8")
        log(f"rfdetr {split}: {len(images)} images, {len(anns)} boxes")
    return root


def rfdetr_mapped_checkpoint(a, names: list[str], em) -> Path:
    """The base checkpoint with its 91-wide class head (COCO category ids) reordered so that row i
    is the run's class i: rfdetr keeps the first `classes + 1` rows when the class count differs,
    which would otherwise give class 0 the head of category 0 (unused), class 1 person's, ... The
    last kept row (rfdetr's extra logit) starts from category 0."""
    import torch

    ck = torch.load(str(a.weights), map_location="cpu", weights_only=False)
    sd = ck.get("model") if isinstance(ck, dict) else None
    bias = sd.get("class_embed.bias") if isinstance(sd, dict) else None
    if bias is None or bias.shape[0] != 91:
        log("rfdetr: base checkpoint head is not the 91-wide COCO head; rfdetr initializes it")
        return a.weights
    coco_rows = head_rows(names, em.COCO80)
    rows = [COCO80_CATEGORY_IDS[r] if r is not None else None for r in coco_rows]
    k = len(names)
    changed = 0
    for key in list(sd.keys()):
        t = sd[key]
        # class_embed.{weight,bias} and the two-stage transformer.enc_out_class_embed.<g>.*
        if "class_embed" not in key:
            continue
        if not hasattr(t, "shape") or t.dim() == 0 or t.shape[0] != 91:
            continue
        new = t.clone()
        for i, j in enumerate(rows):
            new[i] = t[j] if j is not None else t[0]
        new[k] = t[0]
        sd[key] = new
        changed += 1
    out = a.work / "rfdetr-base-mapped.pth"
    torch.save(ck, str(out))
    report_head(names, rows, changed)
    return out


def train_rfdetr(a, device: str, names: list[str], em) -> dict:
    import numpy as np
    import torch
    import rfdetr
    from pytorch_lightning.callbacks import Callback

    cls_name = RFDETR_CLASSES.get(a.size)
    if cls_name is None:
        fail(f"unknown RF-DETR size {a.size!r} (nano, small, medium, base, large)")
    data = rfdetr_layout(a, names)
    started = time.time()

    class Progress(Callback):
        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            nb = trainer.num_training_batches
            i = batch_idx + 1
            if isinstance(nb, int) and nb > 0 and (i == nb or i % max(1, nb // 10) == 0):
                event("batch", epoch=trainer.current_epoch + 1, epochs=trainer.max_epochs, batch=i, batches=nb)

        def on_train_epoch_end(self, trainer, pl_module):
            m = {k: float(v) for k, v in trainer.callback_metrics.items() if getattr(v, "numel", lambda: 0)() == 1}
            loss = next((m[k] for k in ("train/loss", "train/loss_epoch", "loss") if k in m), None)
            mp = next((m[k] for k in ("val/mAP_50", "val/ema_mAP_50", "val/map50") if k in m), None)
            event("epoch", epoch=trainer.current_epoch + 1, epochs=trainer.max_epochs, loss=loss, map50=mp,
                  elapsed_s=round(time.time() - started, 1))

    # rfdetr builds its Lightning trainer itself; add the progress callback to every trainer it builds.
    import pytorch_lightning as pl

    orig_init = pl.Trainer.__init__

    def init(self, *args, **kw):
        cbs = list(kw.get("callbacks") or [])
        cbs.append(Progress())
        kw["callbacks"] = cbs
        orig_init(self, *args, **kw)

    pl.Trainer.__init__ = init
    pretrain = rfdetr_mapped_checkpoint(a, names, em)
    model = getattr(rfdetr, cls_name)(pretrain_weights=str(pretrain))
    out_dir = a.work / "rfdetr-out"
    kw = dict(
        dataset_dir=str(data), epochs=a.epochs, batch_size=a.batch, grad_accum_steps=max(1, 16 // a.batch),
        output_dir=str(out_dir), device=device,
        num_workers=a.workers, resolution=a.imgsz, seed=a.seed,
    )
    if a.lr is not None:
        kw["lr"] = a.lr
    if a.patience:
        kw["early_stopping"] = True
        kw["early_stopping_patience"] = a.patience
    log(f"rfdetr {cls_name}.train({', '.join(f'{k}={v}' for k, v in kw.items())})")
    try:
        model.train(**kw)
    except TypeError as e:
        # A keyword this rfdetr version does not take: retry without the optional ones.
        log(f"rfdetr rejected an option ({e}); retrying with the basic options")
        for k in ("seed", "early_stopping", "early_stopping_patience"):
            kw.pop(k, None)
        model.train(**kw)
    pl.Trainer.__init__ = orig_init
    ckpt = out_dir / "checkpoint_best_total.pth"
    cfg = out_dir / "training_config.json"
    if not ckpt.is_file() or not cfg.is_file():
        fail(f"rfdetr finished without {ckpt.name} / {cfg.name} in {out_dir}")
    tc = json.loads(cfg.read_text(encoding="utf-8"))
    if tc.get("class_names") != names:
        fail(f"rfdetr trained on classes {tc.get('class_names')}, expected {names}")
    event("phase", phase="export", detail="exporting checkpoint_best_total.pth to ONNX")
    wdir = a.work / "rfdetr-export-weights"
    wdir.mkdir(parents=True, exist_ok=True)
    link_or_copy(ckpt, wdir / ckpt.name)
    (wdir / "config.json").write_text(json.dumps({"model_config": tc["model_config"], "class_names": names}), encoding="utf-8")
    ex = argparse.Namespace(
        weights=wdir, work=a.work / "export", out=a.out_dir, opset=a.opset, no_onnx=False, no_openvino=True,
        fp32=False, custom_classes=True,
    )
    ex.work.mkdir(parents=True, exist_ok=True)
    out = em.export_rfdetr(a.name, "rfdetr", a.imgsz, ex)
    # The export goes through transformers' RfDetrForObjectDetection: compare it with rfdetr's own model.
    native = getattr(rfdetr, cls_name)(pretrain_weights=str(ckpt), num_classes=len(names))
    net = native.model.model.eval().float().cpu()
    x = em.synthetic_image(a.imgsz)
    mean = np.array(em.IMAGENET_MEAN, dtype=np.float32)[None, :, None, None]
    std = np.array(em.IMAGENET_STD, dtype=np.float32)[None, :, None, None]
    x = ((x - mean) / std).astype(np.float32)
    with torch.no_grad():
        ref = net(torch.from_numpy(x))
    slice_logits(a.name, out["onnx"], len(names))
    em.stamp_coreml_cache_key(out["onnx"])
    got = em.compare_with_onnxruntime(a.name, out["onnx"], {"pixel_values": x}, {})
    if got["logits"].shape[-1] != len(names):
        fail(f"{a.name}: logits have {got['logits'].shape[-1]} classes after the cut, expected {len(names)}")
    sig = lambda v: 1 / (1 + np.exp(-v))
    ref_named = ref["pred_logits"].numpy()[..., : len(names)]
    diff = float(np.abs(np.sort(sig(ref_named).ravel())[-20:] - np.sort(sig(got["logits"]).ravel())[-20:]).max())
    log(f"{a.name}: top-20 score difference rfdetr PyTorch vs ONNX Runtime: {diff:.2e}")
    if diff > 1e-2:
        fail(f"{a.name}: the ONNX export and rfdetr's PyTorch model disagree (top scores differ by {diff:.3f})")
    return {"family": "rfdetr", **out}


# ---------------------------------------------------------------------------------------------
# transformers (RT-DETRv2, D-FINE)
# ---------------------------------------------------------------------------------------------


def query_box_difference(ref_scores, ref_boxes, got_scores, got_boxes, k: int = 20) -> float:
    """Largest box difference (normalized units) between the `k` best PyTorch queries and their
    counterparts in the ONNX Runtime output. The two-stage query selection is a top-k over the
    encoder's scores, which a freshly trained head leaves nearly tied, so the runtimes may list the
    same queries in another order: each reference query is matched to the ONNX query with the
    closest class scores and box, not to the same index."""
    import numpy as np

    top = np.argsort(-ref_scores.max(-1))[:k]
    worst = 0.0
    for q in top:
        d_score = np.abs(got_scores - ref_scores[q]).max(-1)
        d_box = np.abs(got_boxes - ref_boxes[q]).max(-1)
        worst = max(worst, float((d_box + (d_score > 1e-2) * 1e9).min()))
    return worst


class CocoDataset:
    """The prepared COCO JSON as a torch dataset for transformers' image processors: random
    horizontal flip and night augmentation on the training split."""

    def __init__(self, data: Path, split: str, processor, train: bool, a):
        import torch

        self.data, self.processor, self.train, self.a = data, processor, train, a
        coco = json.loads((data / f"{split}.coco.json").read_text(encoding="utf-8"))
        anns: dict[int, list] = {}
        for x in coco["annotations"]:
            anns.setdefault(x["image_id"], []).append(x)
        self.items = [(im, anns.get(im["id"], [])) for im in coco["images"]]
        self.gen = torch.Generator().manual_seed(a.seed + (0 if train else 1))

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        import numpy as np
        import torch
        from PIL import Image

        im, anns = self.items[i]
        with Image.open(self.data / im["file_name"]) as pil:
            arr = np.asarray(pil.convert("RGB"))
        h, w = arr.shape[:2]
        boxes = [list(map(float, x["bbox"])) for x in anns]
        if self.train:
            if torch.rand(1, generator=self.gen).item() < 0.5:
                arr = arr[:, ::-1].copy()
                boxes = [[w - bx - bw, by, bw, bh] for bx, by, bw, bh in boxes]
            if any_augment(self.a):
                x = torch.from_numpy(arr.astype(np.float32) / 255).permute(2, 0, 1)
                y = night_augment(x, self.a.aug_dark, self.a.aug_noise, self.a.aug_gray, gen=self.gen)
                arr = (y.permute(1, 2, 0).numpy() * 255 + 0.5).clip(0, 255).astype(np.uint8)
        target = {
            "image_id": int(im["id"]),
            "annotations": [
                {"image_id": int(im["id"]), "category_id": int(x["category_id"]), "bbox": b, "area": b[2] * b[3], "iscrowd": 0}
                for x, b in zip(anns, boxes)
                if b[2] > 1 and b[3] > 1
            ],
        }
        out = self.processor(images=arr, annotations=target, return_tensors="pt")
        return {"pixel_values": out["pixel_values"][0], "labels": out["labels"][0]}


def collate(batch):
    import torch

    return {"pixel_values": torch.stack([b["pixel_values"] for b in batch]), "labels": [b["labels"] for b in batch]}


def train_transformers(a, device: str, names: list[str], em) -> dict:
    import numpy as np
    import torch
    from transformers import AutoImageProcessor, AutoModelForObjectDetection, Trainer, TrainerCallback, TrainingArguments

    wdir = a.weights
    for f in ("config.json", "model.safetensors", "preprocessor_config.json"):
        if not (wdir / f).is_file():
            fail(f"{wdir / f} is missing")
    proc = AutoImageProcessor.from_pretrained(str(wdir), local_files_only=True)
    proc.size = {"height": a.imgsz, "width": a.imgsz}
    if getattr(proc, "do_normalize", False):
        fail(f"{wdir}: the image processor normalizes; the service's `detr` family feeds RGB 0..1")
    id2label = dict(enumerate(names))
    model = AutoModelForObjectDetection.from_pretrained(
        str(wdir), id2label=id2label, label2id={n: i for i, n in id2label.items()}, ignore_mismatched_sizes=True,
        local_files_only=True,
    )
    base_cfg = json.loads((wdir / "config.json").read_text(encoding="utf-8"))
    base_labels = base_cfg.get("id2label") or {}
    base_names = [base_labels.get(str(i), "") for i in range(len(base_labels))]
    if base_names and not (len(base_names) == len(names) and head_rows(names, base_names) == list(range(len(names)))):
        from safetensors.torch import load_file

        rows = head_rows(names, base_names)
        changed = copy_head_rows(model.state_dict(), load_file(str(wdir / "model.safetensors")), rows, len(base_names))
        report_head(names, rows, changed)
    kind = type(model).__name__
    want = {"rtdetrv2": "RTDetrV2ForObjectDetection", "dfine": "DFineForObjectDetection"}[a.family]
    if kind != want:
        fail(f"{wdir} holds a {kind}, expected {want}")
    train_ds = CocoDataset(a.data, "train", proc, True, a)
    val_ds = CocoDataset(a.data, "valid", proc, False, a)
    started = time.time()
    steps_per_epoch = max(1, math.ceil(len(train_ds) / a.batch))

    class Progress(TrainerCallback):
        def on_step_end(self, args, state, control, **kw):
            i = ((state.global_step - 1) % steps_per_epoch) + 1
            if i == steps_per_epoch or i % max(1, steps_per_epoch // 10) == 0:
                event("batch", epoch=math.ceil(state.global_step / steps_per_epoch), epochs=a.epochs, batch=i, batches=steps_per_epoch)

        def on_epoch_end(self, args, state, control, **kw):
            losses = [h["loss"] for h in state.log_history if "loss" in h]
            event("epoch", epoch=round(state.epoch or 0), epochs=a.epochs, loss=losses[-1] if losses else None,
                  elapsed_s=round(time.time() - started, 1))

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics and "eval_loss" in metrics:
                event("eval", epoch=round(state.epoch or 0), epochs=a.epochs, val_loss=float(metrics["eval_loss"]))

    targs = TrainingArguments(
        output_dir=str(a.work / "hf-out"),
        num_train_epochs=a.epochs,
        per_device_train_batch_size=a.batch,
        per_device_eval_batch_size=a.batch,
        learning_rate=a.lr if a.lr is not None else 1e-4,
        weight_decay=1e-4,
        max_grad_norm=0.1,
        lr_scheduler_type="cosine",
        warmup_steps=min(100, steps_per_epoch * a.epochs // 10),
        eval_strategy="epoch",
        save_strategy="no",
        logging_strategy="steps",
        logging_steps=max(1, steps_per_epoch // 4),
        remove_unused_columns=False,
        eval_do_concat_batches=False,
        report_to=[],
        dataloader_num_workers=a.workers,
        seed=a.seed,
        use_cpu=device == "cpu",
        fp16=device.startswith("cuda"),
        disable_tqdm=True,
    )
    trainer = Trainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate,
                      callbacks=[Progress()])
    trainer.train()
    event("phase", phase="export", detail="exporting to ONNX")
    model = trainer.model.eval().float().cpu()

    class Wrapped(torch.nn.Module):
        def __init__(self, m):
            super().__init__()
            self.m = m

        def forward(self, pixel_values):
            o = self.m(pixel_values=pixel_values)
            return o.logits, o.pred_boxes

    wrapped = Wrapped(model).eval()
    x = em.synthetic_image(a.imgsz)
    xt = torch.from_numpy(x)
    with torch.no_grad():
        logits, boxes = wrapped(xt)
    work = a.work / "export"
    work.mkdir(parents=True, exist_ok=True)
    onnx = work / f"{a.name}.onnx"
    log(f"{a.name}: exporting ONNX (opset {a.opset}, {a.imgsz}x{a.imgsz})")
    em.torch_export(wrapped, (xt,), ["pixel_values"], ["logits", "pred_boxes"], onnx, a.opset)
    got = em.compare_with_onnxruntime(a.name, onnx, {"pixel_values": x}, {"logits": logits.numpy(), "pred_boxes": boxes.numpy()})
    em.pin_output_shapes(a.name, onnx, got)
    inputs, outputs = em.onnx_ports(onnx)
    # Same ports as the `rfdetr` check (pixel_values -> logits [1,Q,C], pred_boxes [1,Q,4]).
    log(f"{a.name}: " + em.check_ports(a.name, "rfdetr", inputs, outputs, a.imgsz))
    if outputs["logits"][2] != len(names):
        fail(f"{a.name}: logits have {outputs['logits'][2]} classes, the dataset {len(names)}")
    if len(names) == 91:
        fail("91 classes look like a COCO category-id head to the service; use another class count")
    sig = lambda v: 1 / (1 + np.exp(-v))
    diff = float(np.abs(np.sort(sig(got["logits"]).ravel())[-20:] - np.sort(sig(logits.numpy()).ravel())[-20:]).max())
    log(f"{a.name}: top-20 score difference PyTorch vs ONNX Runtime: {diff:.2e}")
    if diff > 1e-2:
        fail(f"{a.name}: ONNX Runtime and PyTorch disagree (top scores differ by {diff:.3f})")
    box_diff = query_box_difference(sig(logits.numpy()[0]), boxes.numpy()[0], sig(got["logits"][0]), got["pred_boxes"][0]) * a.imgsz
    log(f"{a.name}: box difference of the top 20 queries: {box_diff:.3f} px")
    if box_diff > 0.5:
        fail(f"{a.name}: ONNX Runtime and PyTorch boxes disagree by {box_diff:.2f} px")
    out = em.finish_hf(a.name, onnx, argparse.Namespace(out=a.out_dir, no_onnx=False, no_openvino=True, fp32=False), {}, a.imgsz, names)
    return {"family": "detr", **out}


# ---------------------------------------------------------------------------------------------


def main() -> int:
    a = parse_args()
    if a.self_check:
        return self_check(a.trainer, a.device)
    for d in (a.out_dir, a.work):
        d.mkdir(parents=True, exist_ok=True)
    a.data, a.work, a.out_dir, a.weights = a.data.resolve(), a.work.resolve(), a.out_dir.resolve(), a.weights.resolve()
    random.seed(a.seed)
    names = read_classes(a.data)
    train, valid = read_split(a.data, "train"), read_split(a.data, "valid")
    if not train:
        fail("the training split is empty")
    watch_parent()
    em = load_export_module(a.export_script)
    device = pick_device(a.device, a.allow_cpu)
    if device == "mps":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    event("start", trainer=a.trainer, family=a.family, size=a.size, device=device, classes=names,
          train_images=len(train), valid_images=len(valid), epochs=a.epochs, imgsz=a.imgsz, batch=a.batch, **device_info())
    t0 = time.time()
    if a.trainer == "ultralytics":
        out = train_ultralytics(a, device, names, em)
    elif a.trainer == "rfdetr":
        out = train_rfdetr(a, device, names, em)
    else:
        out = train_transformers(a, device, names, em)
    event("done", onnx=str(out["onnx"]), yaml=str(out["yaml"]), family=out["family"], classes=out["names"],
          imgsz=a.imgsz, seconds=round(time.time() - t0, 1), device=device)
    return 0


if __name__ == "__main__":
    sys.exit(main())
