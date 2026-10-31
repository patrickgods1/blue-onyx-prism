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

Progress: every event is one line `@train {json}` on stdout (`start`, `epoch`, `eval`, `batch`,
`stop`, `phase`, `done`); the service parses them, everything else is log output. `epoch` and `eval` carry
`metrics`: every scalar metric of the epoch under the trainer's own key (`val/mAP_50_95`,
`metrics/mAP50(B)`, `eval_loss`, ...) plus this script's `f1`/`f2`/`f0_5` sweeps (`fbeta_sweep`),
`mem_gb` (peak GPU memory) and `img_s` (training images per second); the service maps the keys to
stable ids (src/training/progress.rs). Night augmentation
(`--aug-dark`, `--aug-noise`, `--aug-gray`: per-image probabilities) is applied on the fly to the
training batches (Ultralytics, transformers) or to one extra copy of each training image chosen by
those probabilities (rfdetr).

Convergence (`Convergence`, the same rule for every trainer; their own early stopping is off):
after each validation, the run reaches a stop point once `--min-epochs` passed and `--patience`
epochs brought no gain above `--min-delta` in the mean of the last `--smooth` mAP 50-95 (else
mAP 50) values. While the training loss still falls over those epochs (`loss_falling`: the later
half's mean more than 2% below the earlier half's) it goes on instead, at most twice, each time
for another `--patience` epochs (a `converge_point` event; the service offers Stop now / Continue);
else it stops. It stops at once when `--stop-file` exists (the service's Finish now / Stop now); a
`stop` event says why, and the best (raw mAP) epoch's weights are exported.

Export (`prepare_export`, after training, in this process): the training state is released first
(DataLoader workers stopped, trainer and optimizer dropped, GPU cache emptied), the CPU threads are
capped (`BOP_EXPORT_THREADS`, else all physical cores but one) and the export waits up to 5 min for
3 GB available memory / 4 GB available commit (Windows), then goes ahead anyway.

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


# Extensions past a stop point while the training loss still falls (`Convergence`), and the
# relative drop of the loss that counts as falling (`loss_falling`).
MAX_EXTENSIONS = 2
LOSS_FALL = 0.02


def mean(values) -> float:
    """Left-to-right sum / count (not `sum()`: Python 3.12+ compensates, the Rust twin does not)."""
    s = 0.0
    for v in values:
        s += v
    return s / len(values)


def loss_halves(losses):
    """(mean of the earlier n // 2, mean of the later n // 2) of the finite values of `losses` (None
    and non-finite ones skipped; an odd middle value counts in neither); None below 4 values."""
    v = [float(x) for x in losses if x is not None and math.isfinite(float(x))]
    half = len(v) // 2
    if half < 2:
        return None
    return mean(v[:half]), mean(v[len(v) - half:])


def loss_falling(losses) -> bool:
    """Whether the training loss still falls over `losses` (in epoch order): the later half's mean
    is below the earlier half's by more than LOSS_FALL of it (`loss_halves`; too few values: False).
    The twin of src/training/progress.rs `loss_falling`."""
    h = loss_halves(losses)
    return h is not None and h[1] < h[0] - LOSS_FALL * abs(h[0])


class Convergence:
    """When to stop training (one rule for every trainer; src/training/progress.rs
    `convergence_run` is its twin, parity-tested). `update(epoch, value, loss)` once per validated
    epoch with its mAP (None when there is none) and training loss: "finish" when the stop file
    exists, "converged" once `epoch >= min_epochs` and `patience` epochs passed since the last gain
    above `min_delta` (patience 0 = never), else None. The gain is measured on the mean of the last
    `smooth` mAP values (1 = the raw value), so one noisy peak of a small validation split does not
    start the countdown. At such a stop point, while the training loss still falls over the last
    `patience` epochs (`loss_falling`), at most MAX_EXTENSIONS times and before the last of
    `epochs` (0 = unknown), the run goes on instead: `point` holds the `converge_point` event and
    the countdown restarts there. Else, with `anneal` > 0 epochs and before the last of `epochs`,
    the first stop point that would stop starts an anneal instead: `annealing` holds the `anneal`
    event, training goes on to `anneal_until` = epoch + anneal (at most `epochs`) and stops there
    ("converged"; later stop points are ignored, the stop file still finishes at once). The trainer
    lowers the learning rate over those epochs (`bend_rfdetr_lr`). A repeated or older epoch is
    ignored. `best` / `best_epoch` are the last such gain (smoothed); `top` / `top_epoch` the highest
    raw value, `improved` whether this epoch set it (the checkpoint to keep). Pure, no torch."""

    def __init__(self, patience: int = 0, min_epochs: int = 0, min_delta: float = 0.0, stop_file=None,
                 smooth: int = 1, epochs: int = 0, anneal: int = 0):
        self.patience, self.min_epochs, self.min_delta = int(patience), int(min_epochs), float(min_delta)
        self.smooth, self.epochs, self.anneal = max(1, int(smooth)), int(epochs), max(0, int(anneal))
        self.stop_file = Path(stop_file) if stop_file else None
        self.last = 0
        self.window = []
        self.losses = []
        self.best, self.best_epoch = None, 0
        self.top, self.top_epoch = None, 0
        self.improved = False
        self.extensions, self.restart = 0, 0
        self.falling = False
        self.point = None
        self.anneal_from, self.anneal_until = 0, 0
        self.annealing = None
        self.decision = None

    def update(self, epoch: int, value, loss=None) -> str | None:
        epoch = int(epoch)
        if epoch <= self.last or self.decision is not None:
            return None
        self.last = epoch
        self.improved = False
        self.point = None
        self.annealing = None
        self.losses.append(float(loss) if loss is not None and math.isfinite(float(loss)) else None)
        if value is not None and math.isfinite(float(value)):
            v = float(value)
            self.window = (self.window + [v])[-self.smooth:]
            m = mean(self.window)
            if self.best is None or m > self.best + self.min_delta:
                self.best, self.best_epoch = m, epoch
            if self.top is None or v > self.top:
                self.top, self.top_epoch, self.improved = v, epoch, True
        if self.stop_file is not None and self.stop_file.exists():
            self.decision = "finish"
        elif self.anneal_until:
            if epoch >= self.anneal_until:
                self.decision = "converged"
        elif (self.patience > 0 and self.best is not None and epoch >= self.min_epochs
              and epoch - max(self.best_epoch, self.restart) >= self.patience):
            recent = self.losses[-self.patience:]
            self.falling = loss_falling(recent)
            if self.falling and self.extensions < MAX_EXTENSIONS and (self.epochs <= 0 or epoch < self.epochs):
                self.extensions += 1
                self.restart = epoch
                nxt = epoch + self.patience
                early, late = loss_halves(recent)
                self.point = {"epoch": epoch, "best_epoch": self.top_epoch, "best": self.top, "loss_falling": True,
                              "loss_from": early, "loss_to": late, "extension": self.extensions,
                              "next_check": nxt if self.epochs <= 0 else min(nxt, self.epochs)}
            elif self.anneal > 0 and (self.epochs <= 0 or epoch < self.epochs):
                until = epoch + self.anneal
                self.anneal_from = epoch
                self.anneal_until = until if self.epochs <= 0 else min(until, self.epochs)
                self.annealing = {"epoch": epoch, "until": self.anneal_until, "best_epoch": self.top_epoch,
                                  "best": self.top, "loss_falling": self.falling, "extensions": self.extensions}
            else:
                self.decision = "converged"
        return self.decision

    def step(self, epoch: int, value, loss=None) -> bool:
        """`update`, then its events (`converge_point`, `stop`): True when training stops here."""
        decision = self.update(epoch, value, loss)
        if self.point is not None:
            p = self.point
            log(f"early stop point at epoch {p['epoch']} (best epoch {p['best_epoch']}, mAP {p['best']}), but the "
                f"training loss is still falling (mean {p['loss_from']:.3f} -> {p['loss_to']:.3f}): continuing to epoch "
                f"{p['next_check']} (extension {p['extension']} of {MAX_EXTENSIONS}; Finish now keeps epoch {p['best_epoch']})")
            event("converge_point", **p)
        if self.annealing is not None:
            p = self.annealing
            log(f"early stop point at epoch {p['epoch']} (best epoch {p['best_epoch']}, mAP {p['best']}): lowering the "
                f"learning rate until epoch {p['until']}, then stopping (Finish now keeps the best epoch)")
            event("anneal", **p)
        if decision:
            self.emit()
        return bool(decision)

    def emit(self) -> None:
        """The `stop` event of the decision (the best epoch is the one exported)."""
        if self.decision == "converged" and self.anneal_until:
            why = "lowered the learning rate for %d epochs after the stop point at epoch %d" % (
                self.anneal_until - self.anneal_from, self.anneal_from)
        elif self.decision == "converged":
            why = "no mAP gain above %g for %d epochs" % (self.min_delta, self.patience)
            if self.smooth > 1:
                why += " (mAP averaged over the last %d epochs)" % self.smooth
            if not self.falling:
                why += "; training loss no longer falling"
            elif self.extensions >= MAX_EXTENSIONS:
                why += "; training loss still falling, but %d extensions used" % MAX_EXTENSIONS
            else:
                why += "; training loss still falling, but no epochs left"
        else:
            why = "finish requested"
        log(f"stopping after epoch {self.last}: {why}; best epoch {self.top_epoch} (mAP {self.top})")
        event("stop", reason=self.decision, epoch=self.last, best_epoch=self.top_epoch, best=self.top,
              loss_falling=self.falling, extensions=self.extensions, anneal_until=self.anneal_until or None)


def bend_lambda_lr(scheduler, steps: int, min_factor: float) -> bool:
    """Lower the learning rate of a `LambdaLR`-like `scheduler` (its `lr_lambdas` and `last_epoch`)
    over the next `steps` scheduler steps: each factor f(s) becomes
    min(f(s), lo + (f0 - lo) * (1 + cos(pi * min(1, (s - s0) / steps))) / 2) from the current step
    s0 with f0 = f(s0) and lo = min(min_factor, f0), so it follows a half cosine from f0 down to
    `min_factor` and never rises above the schedule. False (unchanged) without lambdas or steps.
    No torch: tested with a stand-in."""
    lambdas = getattr(scheduler, "lr_lambdas", None)
    if not isinstance(lambdas, list) or not lambdas or int(steps) < 1:
        return False
    s0, steps = int(scheduler.last_epoch), int(steps)

    def bend(orig):
        f0 = float(orig(s0))
        lo = min(float(min_factor), f0)

        def factor(s):
            t = min(1.0, max(0.0, (s - s0) / steps))
            return min(orig(s), lo + (f0 - lo) * 0.5 * (1.0 + math.cos(math.pi * t)))

        return factor

    scheduler.lr_lambdas = [bend(f) for f in lambdas]
    return True


def bend_rfdetr_lr(trainer, epochs: int, steps_per_epoch: int, min_factor: float) -> None:
    """At the start of an anneal of `epochs` epochs: bend the Lightning trainer's LambdaLR (rfdetr's
    managed step / cosine schedule, stepped per optimizer step) down to `min_factor` over them
    (`bend_lambda_lr`). Anything else only logs: the run still anneals, then stops."""
    try:
        import torch

        configs = getattr(trainer, "lr_scheduler_configs", None) or []
        sched = configs[0].scheduler if configs else None
        if not isinstance(sched, torch.optim.lr_scheduler.LambdaLR):
            log(f"anneal: the learning rate could not be lowered (scheduler {type(sched).__name__}); "
                f"training {epochs} more epochs at the schedule's rate, then stopping")
            return
        if getattr(configs[0], "interval", "step") == "epoch":
            steps_per_epoch = 1
        steps = max(1, int(epochs) * int(steps_per_epoch))
        before = [float(f(sched.last_epoch)) for f in sched.lr_lambdas]
        if bend_lambda_lr(sched, steps, min_factor):
            log(f"anneal: learning rate factor {', '.join(f'{f:.3g}' for f in before)} -> {min_factor:g} over "
                f"{epochs} epochs ({steps} scheduler steps from step {sched.last_epoch})")
        else:
            log("anneal: the learning rate could not be lowered (no lr_lambdas); training on, then stopping")
    except Exception as e:  # a changed Lightning / rfdetr internal: anneal-then-stop still applies
        log(f"anneal: the learning rate could not be lowered ({type(e).__name__}: {e}); training on, then stopping")


def convergence(a) -> Convergence:
    return Convergence(a.patience, a.min_epochs, a.min_delta, a.stop_file, a.smooth, a.epochs, a.anneal)


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
    p.add_argument("--lr-encoder", type=float, default=None,
                   help="RF-DETR: the backbone (encoder) learning rate (default: rfdetr's, 1.5e-4)")
    p.add_argument("--lr-scheduler", choices=("cosine", "step"), default="cosine",
                   help="RF-DETR: cosine (decays to --lr-min-factor over --epochs) or step (rfdetr's default: x0.1 after epoch 100)")
    p.add_argument("--lr-min-factor", type=float, default=0.1,
                   help="RF-DETR: the cosine schedule's and the anneal's final learning rate, as a share of --lr")
    p.add_argument("--device", default="auto", help="auto, cuda, cuda:N, mps or cpu")
    p.add_argument("--allow-cpu", action="store_true", help="train on the CPU when no GPU is found (slow; smoke tests)")
    p.add_argument("--workers", type=int, default=2, help="data loader workers")
    p.add_argument("--multi-scale", choices=("on", "off"), default="on",
                   help="RF-DETR: rfdetr's multi-scale training (on: its default, every batch resized up to ~1.4x --imgsz; off: --imgsz only, much less memory)")
    p.add_argument("--cache", choices=("auto", "ram", "off"), default="auto",
                   help="Ultralytics: cache the decoded training images in RAM (auto: when they fit a quarter of the free memory)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--patience", type=int, default=0, help="stop after this many epochs without improvement (0 = off)")
    p.add_argument("--min-epochs", type=int, default=0, help="no convergence stop before this epoch")
    p.add_argument("--min-delta", type=float, default=0.0, help="smallest validation mAP gain that counts as an improvement")
    p.add_argument("--smooth", type=int, default=1, help="the convergence rule's mAP is the mean of this many last epochs (1 = the raw value)")
    p.add_argument("--anneal", type=int, default=0,
                   help="at the first stop point, train this many more epochs (at most --epochs) with a lowered learning rate (RF-DETR), then stop (0 = stop there)")
    p.add_argument("--stop-file", type=Path, default=None, help="finish after the current epoch (keeping the best one) once this file exists")
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
    if a.patience < 0 or a.min_epochs < 0 or not (a.min_delta >= 0 and math.isfinite(a.min_delta)):
        p.error("--patience, --min-epochs and --min-delta must not be negative")
    if a.smooth < 1:
        p.error("--smooth must be at least 1")
    if a.anneal < 0:
        p.error("--anneal must not be negative")
    if not 0.0 <= a.lr_min_factor <= 1.0:
        p.error("--lr-min-factor must be between 0 and 1")
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
# Metrics (per-epoch `metrics` of the epoch / eval events), memory and speed
# ---------------------------------------------------------------------------------------------

MAX_METRICS = 64


def scalar_metrics(d) -> dict:
    """The finite scalar values of `d` (numbers, 0-d / 1-element tensors and arrays) as floats;
    booleans, strings, NaN and inf are dropped."""
    out = {}
    for k, v in (d or {}).items():
        if isinstance(v, (bool, str, bytes)) or not isinstance(k, str):
            continue
        try:
            if hasattr(v, "numel"):  # torch
                if v.numel() != 1:
                    continue
                f = float(v.detach().float().cpu().item())
            elif hasattr(v, "item"):  # numpy
                if getattr(v, "size", 1) != 1:
                    continue
                f = float(v.item())
            else:
                f = float(v)
        except (TypeError, ValueError, RuntimeError):
            continue
        if math.isfinite(f):
            out[k] = f
    return out


def fbeta_sweep(prec, rec, thresholds, betas=(0.5, 1.0, 2.0)) -> dict:
    """Macro F-beta over a confidence sweep: `prec`, `rec` are `[classes, thresholds]` (classes
    with ground truth only), `thresholds` the confidences. For each beta, the per-class
    `(1 + b^2) P R / (b^2 P + R)` (0 where P = R = 0) is averaged over the classes at every
    threshold and the best threshold kept (the lowest on a tie). Returns `f2`, `f2_thr`,
    `f2_precision`, `f2_recall` (macro P / R at that threshold), the same for `f0_5` and `f1`
    (plus `precision` / `recall` at the best F1 threshold)."""
    import numpy as np

    p = np.asarray(prec, dtype=np.float64)
    r = np.asarray(rec, dtype=np.float64)
    t = np.asarray(thresholds, dtype=np.float64)
    if p.ndim != 2 or p.shape != r.shape or p.shape[0] == 0 or p.shape[1] != t.shape[0]:
        return {}
    out = {}
    for b in betas:
        b2 = b * b
        den = b2 * p + r
        f = np.divide((1 + b2) * p * r, den, out=np.zeros_like(den), where=den > 0)
        macro = f.mean(axis=0)
        i = int(np.argmax(macro))
        name = {0.5: "f0_5", 1.0: "f1", 2.0: "f2"}.get(b, f"f{b:g}".replace(".", "_"))
        out[name] = float(macro[i])
        out[f"{name}_thr"] = float(t[i])
        if name == "f1":
            out["precision"] = float(p[:, i].mean())
            out["recall"] = float(r[:, i].mean())
        else:
            out[f"{name}_precision"] = float(p[:, i].mean())
            out[f"{name}_recall"] = float(r[:, i].mean())
    return out


def match_detections(preds, gts, iou_thr: float = 0.5):
    """Per-class matching data for a P/R sweep (COCO-style greedy matching at `iou_thr`, by
    score, each ground truth matched once): `preds` / `gts` are per image lists of
    `(boxes [N,4] xyxy, labels [N], scores [N])` / `(boxes, labels)` numpy arrays. Returns
    `{class: (scores, matched 0/1, ground truth count)}`."""
    import numpy as np

    data: dict[int, list] = {}
    for (pb, pl, ps), (gb, gl) in zip(preds, gts):
        for c in set(np.asarray(pl).tolist()) | set(np.asarray(gl).tolist()):
            d = data.setdefault(int(c), [[], [], 0])
            pi = np.where(pl == c)[0]
            gi = np.where(gl == c)[0]
            d[2] += len(gi)
            order = pi[np.argsort(-ps[pi], kind="stable")]
            used = np.zeros(len(gi), dtype=bool)
            g = gb[gi]
            for j in order:
                m = 0
                if len(gi):
                    b = pb[j]
                    x1 = np.maximum(b[0], g[:, 0]); y1 = np.maximum(b[1], g[:, 1])
                    x2 = np.minimum(b[2], g[:, 2]); y2 = np.minimum(b[3], g[:, 3])
                    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
                    union = (b[2] - b[0]) * (b[3] - b[1]) + (g[:, 2] - g[:, 0]) * (g[:, 3] - g[:, 1]) - inter
                    iou = np.where(union > 0, inter / np.maximum(union, 1e-12), 0.0)
                    iou[used] = -1
                    k = int(np.argmax(iou))
                    if iou[k] >= iou_thr:
                        used[k] = True
                        m = 1
                d[0].append(float(ps[j]))
                d[1].append(m)
    return {c: (np.asarray(v[0]), np.asarray(v[1]), v[2]) for c, v in data.items()}


def pr_sweep(matches: dict, thresholds):
    """Per-class precision and recall at each threshold (`[classes with GT, thresholds]`) from
    [`match_detections`]: a detection counts when its score >= the threshold; P = 0 without any
    detection, as rfdetr's sweep."""
    import numpy as np

    t = np.asarray(thresholds, dtype=np.float64)
    ps, rs = [], []
    for c in sorted(matches):
        scores, matched, n_gt = matches[c]
        if n_gt <= 0:
            continue
        above = scores[None, :] >= t[:, None] if len(scores) else np.zeros((len(t), 0), dtype=bool)
        tp = (above & (matched[None, :] == 1)).sum(axis=1).astype(np.float64)
        n = above.sum(axis=1).astype(np.float64)
        ps.append(np.divide(tp, n, out=np.zeros_like(tp), where=n > 0))
        rs.append(tp / n_gt)
    return np.array(ps), np.array(rs)


class EpochMeter:
    """Peak GPU memory (`mem_gb`) and training speed (`img_s`) of one epoch. MPS has no peak
    counter: the driver's allocation is sampled after every batch and the largest kept."""

    def __init__(self, device: str):
        self.device = device
        self.start = time.time()
        self.train_end = None
        self.peak = 0.0
        self.images = 0

    def epoch_start(self):
        import torch

        self.start, self.train_end, self.peak, self.images = time.time(), None, 0.0, 0
        if self.device.startswith("cuda"):
            try:
                torch.cuda.reset_peak_memory_stats(self.device)
            except Exception:
                pass

    def batch_end(self, images: int):
        self.images += max(0, int(images))
        if self.device == "mps":
            import torch

            try:
                self.peak = max(self.peak, float(torch.mps.driver_allocated_memory()))
            except Exception:
                pass

    def train_done(self):
        if self.train_end is None:
            self.train_end = time.time()

    def metrics(self, images: int | None = None) -> dict:
        import torch

        out = {}
        mem = self.peak
        if self.device.startswith("cuda"):
            try:
                mem = float(torch.cuda.max_memory_reserved(self.device))
            except Exception:
                mem = 0.0
        if mem > 0:
            out["mem_gb"] = round(mem / 2**30, 3)
        end = self.train_end or time.time()
        n = images if images else self.images
        if n and end > self.start:
            out["img_s"] = round(n / (end - self.start), 3)
        return out


def free_memory(device: str) -> None:
    """Return cached GPU memory after a validation (its activations are not reused by training)."""
    import gc

    import torch

    gc.collect()
    try:
        if device == "mps":
            torch.mps.empty_cache()
        elif device.startswith("cuda"):
            torch.cuda.empty_cache()
    except Exception:
        pass


def cuda_capability(device: str):
    import torch

    if not device.startswith("cuda") or not torch.cuda.is_available():
        return None
    try:
        return torch.cuda.get_device_capability(device)
    except Exception:
        return None


def tune_backend(device: str, fixed_size: bool) -> None:
    """CUDA: TF32 matmuls / convolutions on Ampere and newer (compute capability >= 8.0), and
    cuDNN autotuning when every batch has the same size (`fixed_size`, Ultralytics)."""
    import torch

    cc = cuda_capability(device)
    if cc is None:
        return
    if cc[0] >= 8:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
        log(f"CUDA compute capability {cc[0]}.{cc[1]}: TF32 matmuls on")
    if fixed_size:
        torch.backends.cudnn.benchmark = True


def emit_metrics(kind: str, metrics: dict, **fields) -> None:
    """An `epoch` / `eval` event with at most MAX_METRICS metrics (the shortest keys first, so a
    long tail of per-class keys is what is cut)."""
    m = scalar_metrics(metrics)
    if len(m) > MAX_METRICS:
        keep = sorted(m, key=lambda k: (k.count("/") > 1, len(k), k))[:MAX_METRICS]
        m = {k: m[k] for k in keep}
    event(kind, **fields, metrics=m)


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


def free_ram_bytes() -> int | None:
    """Memory the OS can hand out now (free + inactive / available), if known."""
    try:
        if sys.platform == "darwin":
            import subprocess

            out = subprocess.run(["vm_stat"], capture_output=True, text=True, timeout=10).stdout
            page = 16384
            pages = 0
            for line in out.splitlines():
                if "page size of" in line:
                    page = int(line.split("page size of")[1].split()[0])
                for key in ("Pages free", "Pages inactive", "Pages speculative", "Pages purgeable"):
                    if line.startswith(key + ":"):
                        pages += int(line.split(":")[1].strip().rstrip("."))
            return pages * page
        if hasattr(os, "sysconf") and "SC_AVPHYS_PAGES" in os.sysconf_names:
            meminfo = Path("/proc/meminfo")
            if meminfo.is_file():
                for line in meminfo.read_text().splitlines():
                    if line.startswith("MemAvailable:"):
                        return int(line.split()[1]) * 1024
            return os.sysconf("SC_AVPHYS_PAGES") * os.sysconf("SC_PAGE_SIZE")
        if os.name == "nt":
            st = windows_memory_status()
            return st[0] if st else None
    except Exception:
        return None
    return None


def windows_memory_status():
    """(available physical memory, available commit) in bytes from GlobalMemoryStatusEx, or None.
    Available commit (`ullAvailPageFile`) is what new allocations may still reserve: RAM plus the
    paging file minus what is committed."""
    if os.name != "nt":
        return None
    try:
        import ctypes

        class MS(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ulong), ("load", ctypes.c_ulong), ("total", ctypes.c_ulonglong),
                        ("avail", ctypes.c_ulonglong), ("tp", ctypes.c_ulonglong), ("ap", ctypes.c_ulonglong),
                        ("tv", ctypes.c_ulonglong), ("av", ctypes.c_ulonglong), ("ax", ctypes.c_ulonglong)]

        ms = MS()
        ms.len = ctypes.sizeof(MS)
        if ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms)):
            return int(ms.avail), int(ms.ap)
    except Exception:
        return None
    return None


# ---------------------------------------------------------------------------------------------
# System load around the export (after training, in the same process)
# ---------------------------------------------------------------------------------------------

# The export (a PyTorch trace, then an ONNX Runtime session) waits for this much available
# physical memory and (Windows) available commit, at most MEMORY_WAIT_S, polling every
# MEMORY_POLL_S; then it goes ahead anyway. On a 16 GB machine that already paged, starting the
# export at once thrashed the paging file until the machine stopped responding.
EXPORT_MIN_RAM = 3 << 30
EXPORT_MIN_COMMIT = 4 << 30
MEMORY_WAIT_S = 300
MEMORY_POLL_S = 10


def gib(n) -> str:
    return "?" if n is None else f"{n / (1 << 30):.1f} GB"


def memory_status() -> dict:
    """Available physical memory (`ram`) and, on Windows, available commit (`commit`), in bytes;
    None where unknown."""
    win = windows_memory_status()
    if win:
        return {"ram": win[0], "commit": win[1]}
    return {"ram": free_ram_bytes(), "commit": None}


def memory_shortfall(ram, commit, min_ram: int = EXPORT_MIN_RAM, min_commit: int = EXPORT_MIN_COMMIT) -> list:
    """What is short for the export, e.g. ["available memory 1.2 GB < 3.0 GB"]; [] = enough. An
    unknown value (None) never counts as short."""
    short = []
    if ram is not None and ram < min_ram:
        short.append(f"available memory {gib(ram)} < {gib(min_ram)}")
    if commit is not None and commit < min_commit:
        short.append(f"available commit (RAM + paging file) {gib(commit)} < {gib(min_commit)}")
    return short


def wait_for_memory(stop_file=None, wait_s: float = MEMORY_WAIT_S, poll_s: float = MEMORY_POLL_S,
                    status=memory_status, sleep=time.sleep, clock=time.monotonic) -> bool:
    """Wait until `memory_shortfall` is empty, at most `wait_s` seconds. True when there is enough
    memory (or it is unknown); False when the wait ran out or a stop file appeared during it: the
    export then goes ahead anyway (never fail just for this). A stop file that already existed
    (Finish now ended training) does not end the wait. Never raises."""
    try:
        st = status()
        short = memory_shortfall(st.get("ram"), st.get("commit"))
        if not short:
            log(f"memory before the export: {gib(st.get('ram'))} available, commit {gib(st.get('commit'))} available")
            return True
        stop = Path(stop_file) if stop_file else None
        stop_before = stop is not None and stop.exists()
        detail = "low memory (" + "; ".join(short) + f"): waiting up to {round(wait_s / 60)} min for memory to free"
        log("warning: " + detail + "; closing other programs speeds this up")
        event("phase", phase="export", detail=detail)
        until = clock() + wait_s
        while clock() < until:
            sleep(min(poll_s, max(0.0, until - clock())))
            st = status()
            short = memory_shortfall(st.get("ram"), st.get("commit"))
            if not short:
                log(f"memory freed: {gib(st.get('ram'))} available, commit {gib(st.get('commit'))} available")
                event("phase", phase="export", detail="exporting to ONNX")
                return True
            if stop is not None and not stop_before and stop.exists():
                log("stop requested while waiting for memory; exporting now")
                break
        log("warning: still " + "; ".join(short) + "; exporting anyway (the machine may page heavily)")
        event("phase", phase="export", detail="exporting to ONNX (low memory)")
        return False
    except Exception as e:
        log(f"memory check skipped ({type(e).__name__}: {e})")
        return True


def process_memory():
    """This process's resident memory (`rss`), private commit (`private`, Windows) and its child
    processes (`children`, `children_rss`: DataLoader workers) in bytes, via psutil; None without
    psutil. Never raises."""
    try:
        import psutil

        me = psutil.Process()
        mi = me.memory_info()
        out = {"rss": mi.rss, "private": getattr(mi, "private", None)}
        kids = me.children(recursive=True)
        rss = 0
        for k in kids:
            try:
                rss += k.memory_info().rss
            except Exception:
                pass
        out["children"] = len(kids)
        out["children_rss"] = rss
        return out
    except Exception:
        return None


def describe_process_memory(m) -> str:
    if not m:
        return "?"
    text = f"RSS {gib(m['rss'])}"
    if m.get("private") is not None:
        text += f", commit {gib(m['private'])}"
    return text + f", {m['children']} child processes ({gib(m['children_rss'])})"


def shutdown_dataloader_workers() -> int:
    """Stop the worker processes of every live multi-process DataLoader iterator (persistent
    workers outlive training: Lightning's fit loop and Ultralytics' InfiniteDataLoader keep their
    iterators) and drop the loaders' cached iterators. Returns the number of iterators stopped."""
    import gc

    from torch.utils.data import DataLoader
    from torch.utils.data import dataloader as dl

    cls = getattr(dl, "_MultiProcessingDataLoaderIter", None)
    if cls is None:
        return 0
    stopped = 0
    loaders = []
    for obj in gc.get_objects():
        try:
            # type(), not isinstance(): the latter reads `__class__`, which deprecated module
            # proxies (torch.distributed.reduce_op) answer with a FutureWarning.
            t = type(obj)
            if issubclass(t, cls):
                if not getattr(obj, "_shutdown", True):
                    obj._shutdown_workers()
                    stopped += 1
            elif issubclass(t, DataLoader):
                loaders.append(obj)
        except Exception:
            pass
    for loader in loaders:
        try:
            if getattr(loader, "_iterator", None) is not None:
                loader._iterator = None
        except Exception:
            pass
    return stopped


def release_training_memory(device: str) -> None:
    """After training, before the export: stop the DataLoader workers, collect the training state
    the caller dropped (trainer, optimizer, datasets), return cached GPU memory. Logs the process's
    memory before -> after (psutil). Never raises."""
    try:
        import gc

        before = process_memory()
        gc.collect()
        try:
            workers = shutdown_dataloader_workers()
        except Exception as e:
            workers = 0
            log(f"DataLoader workers not stopped ({type(e).__name__}: {e})")
        gc.collect()
        free_memory(device)
        after = process_memory()
        log(f"released the training state: {workers} DataLoader iterator(s) stopped; "
            f"{describe_process_memory(before)} -> {describe_process_memory(after)}")
    except Exception as e:
        log(f"releasing the training memory failed ({type(e).__name__}: {e})")


def export_threads() -> int:
    """CPU threads for the export: `BOP_EXPORT_THREADS` (set by the service), else all physical
    cores but one, at least one."""
    try:
        n = int(os.environ.get("BOP_EXPORT_THREADS", ""))
        if n > 0:
            return n
    except ValueError:
        pass
    cores = None
    try:
        import psutil

        cores = psutil.cpu_count(logical=False)
    except Exception:
        pass
    return max(1, (cores or os.cpu_count() or 2) - 1)


def prepare_export(device: str, stop_file=None) -> None:
    """Between training and export (the caller dropped its references to the training state
    first): release that memory, cap the CPU threads of the trace and the ONNX Runtime check
    (`BOP_EXPORT_THREADS`, also read by the export script), wait for enough free memory. Never
    raises."""
    release_training_memory(device)
    n = export_threads()
    os.environ["BOP_EXPORT_THREADS"] = str(n)
    try:
        import torch

        torch.set_num_threads(n)
        log(f"export: {n} CPU threads")
    except Exception as e:
        log(f"export thread cap skipped ({type(e).__name__}: {e})")
    wait_for_memory(stop_file)


def ultralytics_cache(a):
    """`cache="ram"` when the decoded training images (unique images x imgsz^2 x 3 bytes, the
    letterboxed size Ultralytics keeps) fit a quarter of the free memory, else None (no cache)."""
    if a.cache == "off":
        return None
    unique = len(set(read_split(a.data, "train")))
    need = unique * a.imgsz * a.imgsz * 3
    if a.cache == "ram":
        return "ram"
    free = free_ram_bytes()
    if free is None or need > free / 4:
        log(f"image cache off: {unique} images need {need / 2**30:.1f} GB, a quarter of the free memory is "
            f"{(free or 0) / 4 / 2**30:.1f} GB")
        return None
    log(f"image cache in RAM: {unique} images, {need / 2**30:.2f} GB")
    return "ram"


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
    meter = EpochMeter(device)

    def on_train_epoch_start(trainer):
        state["batches"] = 0
        state["epoch_started"] = time.time()
        meter.epoch_start()

    def on_train_batch_end(trainer):
        state["batches"] += 1
        meter.batch_end(0)
        nb = len(trainer.train_loader)
        if state["batches"] == nb or state["batches"] % max(1, nb // 10) == 0:
            event("batch", epoch=trainer.epoch + 1, epochs=trainer.epochs, batch=state["batches"], batches=nb)

    def on_train_epoch_end(trainer):
        meter.train_done()

    conv = convergence(a)

    def on_fit_epoch_end(trainer):
        # Also called once more by the final validation of best.pt (epoch + 1, after the loop):
        # report each epoch once, nothing after a convergence stop.
        if trainer.epoch + 1 <= state.get("reported", 0) or trainer.epoch + 1 > trainer.epochs or conv.decision:
            return
        state["reported"] = trainer.epoch + 1
        m = scalar_metrics(trainer.metrics)
        tl = getattr(trainer, "tloss", None)
        if isinstance(tl, dict):
            loss = float(sum(float(v) for v in tl.values()))
            m.update(scalar_metrics(trainer.label_loss_items(tl, prefix="train")))
        elif tl is not None:
            loss = float(tl.sum())
        else:
            loss = None
        if loss is not None and math.isfinite(loss):
            m["loss"] = loss
        val_losses = [v for k, v in m.items() if k.startswith("val/") and k.endswith("_loss")]
        if val_losses:
            m["val_loss"] = float(sum(val_losses))
        m.update(scalar_metrics(getattr(trainer, "lr", None)))
        box = getattr(getattr(getattr(trainer, "validator", None), "metrics", None), "box", None)
        try:
            if box is not None and len(getattr(box, "p_curve", [])):
                m.update(fbeta_sweep(box.p_curve, box.r_curve, box.px))
                # Ultralytics' own P / R (at its max-F1 confidence) are the ones reported.
                m.pop("precision", None)
                m.pop("recall", None)
        except Exception as e:  # a changed Ultralytics internal: the other metrics still count
            log(f"F-beta sweep skipped ({type(e).__name__}: {e})")
        try:
            images = len(trainer.train_loader.dataset)
        except Exception:
            images = None
        m.update(meter.metrics(images))
        emit_metrics(
            "epoch",
            m,
            epoch=trainer.epoch + 1,
            epochs=trainer.epochs,
            loss=loss,
            map50=m.get("metrics/mAP50(B)"),
            map=m.get("metrics/mAP50-95(B)"),
            elapsed_s=round(time.time() - started, 1),
        )
        # Runs after the epoch's checkpoints were saved and before the loop's `if self.stop:
        # break` (ultralytics 8.4 engine/trainer.py `_do_train`): stop ends training here, then
        # best.pt is validated and exported as always.
        tm = getattr(trainer, "metrics", None) or {}
        value = tm.get("metrics/mAP50-95(B)", tm.get("metrics/mAP50(B)"))
        if conv.step(trainer.epoch + 1, value, loss):
            trainer.stop = True

    model.add_callback("on_train_epoch_start", on_train_epoch_start)
    model.add_callback("on_train_batch_end", on_train_batch_end)
    model.add_callback("on_train_epoch_end", on_train_epoch_end)
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
        # Ultralytics' own early stopping off (Convergence decides; it would stop at epochs + 1).
        patience=a.epochs + 1,
        trainer=NightTrainer,
        verbose=True,
    )
    if a.lr is not None:
        kwargs["lr0"] = a.lr
        kwargs["optimizer"] = "AdamW"
    cache = ultralytics_cache(a)
    if cache:
        kwargs["cache"] = cache
    tune_backend(device, fixed_size=True)
    model.train(**kwargs)
    best = a.work / "run" / "weights" / "best.pt"
    if not best.is_file():
        best = a.work / "run" / "weights" / "last.pt"
    if not best.is_file():
        fail("Ultralytics finished without writing best.pt or last.pt")
    event("phase", phase="export", detail=f"exporting {best.name} to ONNX")
    # The export loads best.pt afresh: drop the trained model, its trainer (optimizer, EMA,
    # dataloaders) and the base model first.
    del model, base_yolo, base_model
    prepare_export(device, a.stop_file)
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
    augmentation, the training split gets one augmented copy of each unique image the probabilities
    pick: a frame repeated by oversampling or balancing (several COCO entries of one file) is
    augmented once, so the repeats do not multiply its augmented copies."""
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
        augmented: set[str] = set()
        for im in coco["images"]:
            src = a.data / im["file_name"]
            # One file per source image; repeated entries (oversampling) share it.
            flat = Path(im["file_name"]).name
            link_or_copy(src, d / flat)
            images.append({**im, "file_name": flat})
            if split == "train" and any_augment(a) and flat not in augmented:
                augmented.add(flat)
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
    meter = EpochMeter(device)

    # rfdetr's validation sweeps confidence thresholds for its macro F1 (per-class precision and
    # recall at np.linspace(0, 1, 101)); keep each sweep for the F2 / F0.5 of the same rows.
    import rfdetr.training.callbacks.coco_eval as coco_eval

    sweeps = {"n": 0, "used": 0, "last": {}}
    orig_sweep = getattr(coco_eval, "sweep_confidence_thresholds", None)

    def sweep(per_class_data, conf_thresholds, classes_with_gt, *args, **kw):
        thresholds = list(conf_thresholds)
        res = orig_sweep(per_class_data, thresholds, classes_with_gt, *args, **kw)
        try:
            idx = list(classes_with_gt)
            if res and idx:
                prec = np.array([r["per_class_prec"] for r in res]).T[idx]
                rec = np.array([r["per_class_rec"] for r in res]).T[idx]
                f = fbeta_sweep(prec, rec, [float(r["confidence_threshold"]) for r in res])
                # rfdetr reports F1 and its P / R itself (the same rows, the same best threshold).
                for k in ("f1", "precision", "recall"):
                    f.pop(k, None)
                sweeps["last"] = f
                sweeps["n"] += 1
        except Exception as e:  # a changed rfdetr internal: its own metrics still count
            log(f"F-beta sweep skipped ({type(e).__name__}: {e})")
        return res

    if orig_sweep is not None:
        coco_eval.sweep_confidence_thresholds = sweep

    conv = convergence(a)
    # The scheduler's step at the previous epoch end: its steps per epoch, as they ran.
    sched_steps = {"last": None, "per_epoch": 0}

    def scheduler_step(trainer):
        try:
            return int(trainer.lr_scheduler_configs[0].scheduler.last_epoch)
        except Exception:
            return None

    def steps_per_epoch(trainer) -> int:
        if sched_steps["per_epoch"] > 0:
            return sched_steps["per_epoch"]
        try:
            return max(1, int(trainer.estimated_stepping_batches) // max(1, int(trainer.max_epochs)))
        except Exception:
            return max(1, int(trainer.num_training_batches) // max(1, int(trainer.accumulate_grad_batches)))

    class Progress(Callback):
        def on_train_epoch_start(self, trainer, pl_module):
            meter.epoch_start()

        def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
            nb = trainer.num_training_batches
            i = batch_idx + 1
            try:
                meter.batch_end(len(batch[1]))
            except Exception:
                meter.batch_end(a.batch)
            if isinstance(nb, int) and nb > 0 and i == nb:
                meter.train_done()
            if isinstance(nb, int) and nb > 0 and (i == nb or i % max(1, nb // 10) == 0):
                event("batch", epoch=trainer.current_epoch + 1, epochs=trainer.max_epochs, batch=i, batches=nb)

        def on_validation_start(self, trainer, pl_module):
            meter.train_done()

        def on_validation_end(self, trainer, pl_module):
            # The first training step after the EMA validation is where MPS ran out of memory.
            free_memory(device)

        def on_train_epoch_end(self, trainer, pl_module):
            # Bare `loss`, `loss_cls`, ... are the progress bar's last-step values; the epoch's are `train/...`.
            m = {k: v for k, v in scalar_metrics(trainer.callback_metrics).items()
                 if not k.endswith("_step") and not (k.startswith("loss") and "/" not in k)}
            loss = next((m[k] for k in ("train/loss", "train/loss_epoch", "loss") if k in m), None)
            mp = next((m[k] for k in ("val/mAP_50", "val/ema_mAP_50", "val/map50") if k in m), None)
            mp95 = next((m[k] for k in ("val/mAP_50_95", "val/ema_mAP_50_95") if k in m), None)
            if sweeps["n"] > sweeps["used"]:
                m.update(sweeps["last"])
                sweeps["used"] = sweeps["n"]
            m.update(meter.metrics())
            emit_metrics("epoch", m, epoch=trainer.current_epoch + 1, epochs=trainer.max_epochs, loss=loss, map50=mp,
                         map=mp95, elapsed_s=round(time.time() - started, 1))
            # Lightning runs the epoch's validation before on_train_epoch_end, so the mAP is this
            # epoch's; should_stop ends fit after it (rfdetr's own early stopping is not enabled).
            s = scheduler_step(trainer)
            if s is not None and sched_steps["last"] is not None and s > sched_steps["last"]:
                sched_steps["per_epoch"] = s - sched_steps["last"]
            sched_steps["last"] = s
            if not trainer.sanity_checking and conv.step(trainer.current_epoch + 1, mp95 if mp95 is not None else mp, loss):
                trainer.should_stop = True
            elif conv.annealing is not None:
                p = conv.annealing
                bend_rfdetr_lr(trainer, p["until"] - p["epoch"], steps_per_epoch(trainer), a.lr_min_factor)

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
    if a.lr_encoder is not None:
        kw["lr_encoder"] = a.lr_encoder
    # rfdetr's default step schedule drops x0.1 only after epoch 100, past where the convergence
    # rule usually stops: cosine decays over the planned epochs (its total steps).
    kw["lr_scheduler"] = a.lr_scheduler
    if a.lr_scheduler == "cosine":
        kw["lr_scheduler_kwargs"] = {"min_factor": a.lr_min_factor}
    if a.multi_scale == "off":
        kw["multi_scale"] = False
    if a.workers > 0:
        kw["persistent_workers"] = True
        if device.startswith("cuda"):
            kw["pin_memory"] = True
    tune_backend(device, fixed_size=False)
    log(f"rfdetr {cls_name}.train({', '.join(f'{k}={v}' for k, v in kw.items())})")
    try:
        model.train(**kw)
    except (TypeError, ValueError) as e:
        # A keyword this rfdetr version does not take (TypeError, or its config's "Unknown
        # parameter(s)" ValueError): retry without the optional ones.
        if isinstance(e, ValueError) and "Unknown parameter" not in str(e):
            raise
        log(f"rfdetr rejected an option ({e}); retrying with the basic options")
        for k in ("seed", "multi_scale", "persistent_workers", "pin_memory", "lr_encoder", "lr_scheduler",
                  "lr_scheduler_kwargs"):
            kw.pop(k, None)
        model.train(**kw)
    pl.Trainer.__init__ = orig_init
    if orig_sweep is not None:
        coco_eval.sweep_confidence_thresholds = orig_sweep
    ckpt = out_dir / "checkpoint_best_total.pth"
    cfg = out_dir / "training_config.json"
    if not ckpt.is_file() or not cfg.is_file():
        fail(f"rfdetr finished without {ckpt.name} / {cfg.name} in {out_dir}")
    tc = json.loads(cfg.read_text(encoding="utf-8"))
    if tc.get("class_names") != names:
        fail(f"rfdetr trained on classes {tc.get('class_names')}, expected {names}")
    event("phase", phase="export", detail="exporting checkpoint_best_total.pth to ONNX")
    # One model at a time from here: the trained one (with its Lightning trainer, optimizer and
    # persistent DataLoader workers) goes before the export loads the checkpoint, the export's
    # PyTorch model before ONNX Runtime checks the file (`release_model`), rfdetr's own model
    # before the comparison session.
    del model
    prepare_export(device, a.stop_file)
    wdir = a.work / "rfdetr-export-weights"
    wdir.mkdir(parents=True, exist_ok=True)
    link_or_copy(ckpt, wdir / ckpt.name)
    (wdir / "config.json").write_text(json.dumps({"model_config": tc["model_config"], "class_names": names}), encoding="utf-8")
    ex = argparse.Namespace(
        weights=wdir, work=a.work / "export", out=a.out_dir, opset=a.opset, no_onnx=False, no_openvino=True,
        fp32=False, custom_classes=True, release_model=True,
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
    ref_logits = ref["pred_logits"].numpy()
    del native, net, ref
    free_memory(device)
    slice_logits(a.name, out["onnx"], len(names))
    em.stamp_coreml_cache_key(out["onnx"])
    got = em.compare_with_onnxruntime(a.name, out["onnx"], {"pixel_values": x}, {})
    if got["logits"].shape[-1] != len(names):
        fail(f"{a.name}: logits have {got['logits'].shape[-1]} classes after the cut, expected {len(names)}")
    sig = lambda v: 1 / (1 + np.exp(-v))
    ref_named = ref_logits[..., : len(names)]
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


def detection_metrics(model, ds, proc, batch: int, imgsz: int) -> dict:
    """mAP 50-95 / 50 / 75 and mAR@100 (torchmetrics, COCO rules) plus the macro P / R / F-beta
    sweep at IoU 0.5 ([`fbeta_sweep`]) of `model` on the prepared validation set `ds`, in the
    model's input pixels (images are stretched to imgsz x imgsz, the labels are normalized)."""
    import numpy as np
    import torch
    from torchmetrics.detection import MeanAveragePrecision

    dev = next(model.parameters()).device
    was_training = model.training
    model.eval()
    mean_ap = MeanAveragePrecision(box_format="xyxy", iou_type="bbox")
    preds, gts = [], []
    try:
        with torch.no_grad():
            for i in range(0, len(ds), max(1, batch)):
                items = [ds[j] for j in range(i, min(len(ds), i + max(1, batch)))]
                pv = torch.stack([it["pixel_values"] for it in items]).to(dev)
                out = model(pixel_values=pv)
                res = proc.post_process_object_detection(out, threshold=0.0, target_sizes=[(imgsz, imgsz)] * len(items))
                for it, r in zip(items, res):
                    lab = it["labels"]
                    cx, cy, w, h = (lab["boxes"].float().cpu() * imgsz).unbind(-1) if len(lab["boxes"]) else (torch.zeros(0),) * 4
                    gb = torch.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], -1) if len(lab["boxes"]) else torch.zeros((0, 4))
                    gl = lab["class_labels"].long().cpu()
                    pb, ps, pl = r["boxes"].float().cpu(), r["scores"].float().cpu(), r["labels"].long().cpu()
                    mean_ap.update([{"boxes": pb, "scores": ps, "labels": pl}], [{"boxes": gb, "labels": gl}])
                    preds.append((pb.numpy(), pl.numpy(), ps.numpy()))
                    gts.append((gb.numpy(), gl.numpy()))
    finally:
        model.train(was_training)
    r = mean_ap.compute()
    m = {"map": r["map"], "map50": r["map_50"], "map75": r["map_75"], "mar": r["mar_100"]}
    m = {k: float(v) for k, v in m.items() if float(v) >= 0}  # torchmetrics: -1 = no ground truth
    thresholds = np.linspace(0, 1, 101)
    prec, rec = pr_sweep(match_detections(preds, gts), thresholds)
    m.update(fbeta_sweep(prec, rec, thresholds))
    return m


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
    meter = EpochMeter(device)

    class Progress(TrainerCallback):
        def on_epoch_begin(self, args, state, control, **kw):
            meter.epoch_start()

        def on_step_end(self, args, state, control, **kw):
            meter.batch_end(0)
            i = ((state.global_step - 1) % steps_per_epoch) + 1
            if i == steps_per_epoch or i % max(1, steps_per_epoch // 10) == 0:
                event("batch", epoch=math.ceil(state.global_step / steps_per_epoch), epochs=a.epochs, batch=i, batches=steps_per_epoch)

        def on_epoch_end(self, args, state, control, **kw):
            meter.train_done()
            losses = [h["loss"] for h in state.log_history if "loss" in h]
            lrs = [h["learning_rate"] for h in state.log_history if "learning_rate" in h]
            m = {"loss": losses[-1]} if losses else {}
            if lrs:
                m["lr"] = lrs[-1]
            m.update(meter.metrics(len(train_ds)))
            emit_metrics("epoch", m, epoch=round(state.epoch or 0), epochs=a.epochs, loss=losses[-1] if losses else None,
                         elapsed_s=round(time.time() - started, 1))

        def on_evaluate(self, args, state, control, metrics=None, **kw):
            if metrics and "eval_loss" in metrics:
                emit_metrics("eval", {k: v for k, v in metrics.items() if k != "epoch" and not k.endswith(("_runtime", "_per_second"))},
                             epoch=round(state.epoch or 0), epochs=a.epochs, val_loss=float(metrics["eval_loss"]))

    conv = convergence(a)
    best_state = {}

    class DetectionTrainer(Trainer):
        """Trainer whose evaluation also scores the detections (mAP, P / R, F-beta), keeps the
        best epoch's weights and stops on convergence (or a finish request)."""

        def evaluate(self, *args, **kw):
            out = super().evaluate(*args, **kw)
            m = {}
            epoch = round(self.state.epoch or 0)
            try:
                m = detection_metrics(self.model, val_ds, proc, a.batch, a.imgsz)
                emit_metrics("eval", m, epoch=epoch, epochs=a.epochs)
            except Exception as e:  # the loss still counts
                log(f"detection metrics skipped ({type(e).__name__}: {e})")
            if epoch >= 1:
                losses = [h["loss"] for h in self.state.log_history if "loss" in h]
                stop = conv.step(epoch, m.get("map", m.get("map50")), losses[-1] if losses else None)
                if conv.improved:
                    best_state["epoch"], best_state["value"] = conv.top_epoch, conv.top
                    best_state["sd"] = {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}
                if stop:
                    # Read after this evaluation by the epoch loop (transformers Trainer `train`).
                    self.control.should_training_stop = True
            free_memory(device)
            return out

    cc = cuda_capability(device)
    bf16 = cc is not None and cc[0] >= 8
    tune_backend(device, fixed_size=False)
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
        dataloader_pin_memory=device.startswith("cuda"),
        dataloader_persistent_workers=a.workers > 0,
        seed=a.seed,
        use_cpu=device == "cpu",
        # bf16 on Ampere and newer (no overflow in the DETR losses), fp16 on older CUDA GPUs.
        bf16=bf16,
        fp16=device.startswith("cuda") and not bf16,
        disable_tqdm=True,
    )
    trainer = DetectionTrainer(model=model, args=targs, train_dataset=train_ds, eval_dataset=val_ds, data_collator=collate,
                               callbacks=[Progress()])
    trainer.train()
    if best_state.get("sd") is not None:
        # The best epoch, not the last one (save_strategy="no": nothing else keeps it).
        trainer.model.load_state_dict(best_state["sd"])
        log(f"exporting epoch {best_state['epoch']} (best mAP {best_state['value']:.4f}); the last epoch was {conv.last}")
    else:
        log("exporting the last epoch (no validation mAP was measured)")
    event("phase", phase="export", detail="exporting to ONNX")
    model = trainer.model.eval().float().cpu()
    # Keep only the model: the best epoch's CPU copy, the optimizer, the scheduler and the
    # dataloaders (accelerate holds references too) go before the export.
    best_state.clear()
    try:
        trainer.accelerator.free_memory()
    except Exception as e:
        log(f"accelerate free_memory skipped ({type(e).__name__}: {e})")
    del trainer
    prepare_export(device, a.stop_file)

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
    if sys.platform == "darwin":
        # MPS may otherwise allocate up to 1.7x the recommended working set (more than the RAM of
        # a 16 GB Mac): an oversized batch then swaps for half an hour before it fails. 1.0 keeps
        # allocations inside physical memory, so it fails in seconds (the service retries with a
        # smaller batch). The low watermark must stay below the high one.
        os.environ.setdefault("PYTORCH_MPS_HIGH_WATERMARK_RATIO", "1.0")
        os.environ.setdefault("PYTORCH_MPS_LOW_WATERMARK_RATIO", "0.8")
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
