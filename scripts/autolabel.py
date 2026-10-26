#!/usr/bin/env python
"""Auto-label dataset frames for review (Blue Onyx Prism, `blue-onyx-prism train autolabel`).

Labelers, per frame (docs/TRAINING.md, "Auto-label"):

  1. MM-Grounding-DINO (open vocabulary, Hugging Face transformers, Apache-2.0): one text prompt per
     class plus its synonyms ("raccoon", "Virginia opossum", ...), batched so each prompt stays under
     the text encoder's token limit; boxes at or above --gdino-threshold (0.2), class-aware NMS at
     --nms-iou (0.6).
  2. RF-DETR L (closed set, the `rfdetr` package, Apache-2.0) for the classes that are COCO classes,
     at or above --rfdetr-threshold (0.3).
  3. Weighted boxes fusion of both (IoU --wbf-iou 0.55, weights 1:1; a class only one labeler can
     report is fused over that labeler alone). Every fused box records which labelers found it
     (`sources`) and whether both did (`agree`).
  4. Checks: Qwen3-VL (--vlm, optional) answers "what is this crop" in one word for every person box
     and every single-labeler box; SigLIP 2 (--siglip, optional) ranks the animal classes on the crop
     with day or night prompts (`species_scores`). Neither deletes a box: the service's triage turns a
     "no" into "review". --vlm-min-score skips the VLM on boxes whose fused score is below it
     (`vlm_skipped: "score"`); --vlm-batch N asks up to N crops of a frame in one `generate` call.
     A reply ends after its first token when that is a whole answer word (one decode step less).

Nothing is decided here: the service reads the JSONL output, triages each box (accept only when
both labelers agree at a score >= 0.6 and no check objects; night person / animal boxes always go to
review) and merges them into the frames' label files as pending `source: auto` boxes.

Input (`--frames`): JSONL, one object per frame: `{"frame": "<rel>", "path": "<image>", "width": W,
"height": H, "scene": "day|night|ir"}`. Output (`--out`): JSONL, one object per frame:
`{"frame": "<rel>", "boxes": [{"label", "bbox": [x0, y0, x1, y1], "score", "sources": {"gdino": s,
"rfdetr": s}, "agree", "vlm": {...}, "species_scores": {...}}], "ms": {...}}` (pixels of the frame).

Progress: one stdout line `@autolabel {json}` per event (`start`, `model`, `frame`, `done`,
`self_check`, `self_test`); everything else is log output. Every finished frame is flushed to
`--out` at once, so a stopped run keeps them; --prefetch decodes the next frames on a thread. The
service runs this file (embedded in the binary) in its training environment
(<data root>/tools/train-env), offline, with weights it downloaded and SHA-256 checked.

    python scripts/autolabel.py --self-test           # pure tests of the fusion code, no torch needed
    python scripts/autolabel.py --self-check          # imports and the device
    python scripts/autolabel.py --frames f.jsonl --out p.jsonl --classes '["person","car"]' \\
        --gdino <dir> --rfdetr <rf-detr-large.pth> [--vlm <Qwen3-VL dir>] [--siglip <SigLIP 2 dir>]
"""

from __future__ import annotations

import argparse
import json
import math
import os
import queue
import sys
import threading
import time
import warnings
from pathlib import Path

# ---------------------------------------------------------------------------------------------
# Events
# ---------------------------------------------------------------------------------------------


def log(msg: str) -> None:
    print(f"[autolabel] {msg}", flush=True)


def event(kind: str, **fields) -> None:
    fields["event"] = kind
    print("\n@autolabel " + json.dumps(fields, separators=(",", ":"), default=str), flush=True)


def fail(msg: str) -> None:
    raise SystemExit(f"[autolabel] error: {msg}")


# ---------------------------------------------------------------------------------------------
# Pure helpers (tested by --self-test)
# ---------------------------------------------------------------------------------------------

# The 80 COCO class names (RF-DETR's label space) and aliases of the service's class names.
COCO_NAMES = {
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
    "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
    "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
    "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
    "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
    "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors",
    "teddy bear", "hair drier", "toothbrush",
}
ALIASES = {"motorbike": "motorcycle", "people": "person", "pedestrian": "person", "bike": "bicycle",
           "sofa": "couch", "tvmonitor": "tv", "aeroplane": "airplane",
           "pottedplant": "potted plant", "diningtable": "dining table"}

# Default synonyms of the default class list (the service passes its own map, `training.synonyms`).
DEFAULT_SYNONYMS = {
    "person": ["person", "pedestrian"],
    "car": ["car", "van", "pickup truck"],
    "truck": ["truck", "lorry"],
    "bicycle": ["bicycle"],
    "motorcycle": ["motorcycle", "scooter"],
    "dog": ["dog"],
    "cat": ["cat"],
    "raccoon": ["raccoon"],
    "opossum": ["opossum", "Virginia opossum"],
    "skunk": ["skunk"],
    "fox": ["fox", "red fox"],
    "deer": ["deer", "white-tailed deer"],
    "squirrel": ["squirrel"],
    "bird": ["bird"],
}

PERSON = {"person"}
VEHICLES = {"car", "truck", "bus", "bicycle", "motorcycle", "train", "boat", "airplane", "van"}
ANIMALS = {"dog", "cat", "raccoon", "opossum", "skunk", "fox", "deer", "squirrel", "bird", "horse", "sheep",
           "cow", "bear", "coyote", "rabbit", "hare", "badger", "boar", "wolf", "elephant", "zebra", "giraffe",
           "animal", "mouse", "rat", "goat", "chicken", "duck", "goose", "owl", "bobcat", "cougar", "moose"}
# The service passes its own map (--categories); these are the defaults.
CATEGORIES: dict = {}


def category(label: str) -> str:
    """The category the VLM answer is checked against: person, vehicle or animal; `object` for
    any other class (no VLM check)."""
    if label in CATEGORIES:
        return CATEGORIES[label]
    if label in PERSON:
        return "person"
    if label in VEHICLES:
        return "vehicle"
    if label in ANIMALS:
        return "animal"
    return "object"


def coco_class(label: str):
    """The COCO name of a service class (with aliases), or None."""
    n = ALIASES.get(label.lower(), label.lower())
    return n if n in COCO_NAMES else None


def iou(a, b) -> float:
    iw = min(a[2], b[2]) - max(a[0], b[0])
    ih = min(a[3], b[3]) - max(a[1], b[1])
    if iw <= 0 or ih <= 0:
        return 0.0
    inter = iw * ih
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def nms(boxes: list, thr: float) -> list:
    """Class-aware greedy NMS over dicts with `label`, `bbox`, `score`: highest score first, a box
    of the same label at IoU > thr of a kept one is dropped."""
    out = []
    for b in sorted(boxes, key=lambda x: -x["score"]):
        if all(k["label"] != b["label"] or iou(k["bbox"], b["bbox"]) <= thr for k in out):
            out.append(b)
    return out


def wbf(per_model: dict, weights: dict, iou_thr: float, participants: dict) -> list:
    """Weighted boxes fusion (Solovyev et al. 2021, `conf_type="avg"`) of `per_model`
    ({model: [{"label", "bbox", "score"}]}), per label.

    Boxes are taken in descending score order; a box joins the first cluster of its label whose
    fused box it overlaps at IoU > iou_thr, else starts one. A cluster's box is the score-weighted
    mean of its members' corners (score x model weight); its score is the weighted mean score,
    times min(models in the cluster, models of the label) / models of the label (weights 1:1:
    min(n, M) / M), so a box only one of two labelers found keeps half its score. `participants`
    maps a label to the models that can report it (default: all); a class only one labeler knows
    is fused over that labeler alone (M = its weight). Returns dicts with `label`, `bbox`, `score`,
    `sources` (best score per model) and `agree` (every participating model is in the cluster).
    """
    labels = sorted({b["label"] for bs in per_model.values() for b in bs})
    out = []
    for label in labels:
        models = participants.get(label) or sorted(per_model)
        total_w = sum(weights.get(m, 1.0) for m in models)
        items = []
        for m in models:
            for b in per_model.get(m, []):
                if b["label"] == label:
                    items.append((b["score"] * weights.get(m, 1.0), b["score"], m, b["bbox"]))
        items.sort(key=lambda t: -t[0])
        clusters = []  # [members, fused_box]
        for it in items:
            best, best_iou = None, iou_thr
            for c in clusters:
                v = iou(c[1], it[3])
                if v > best_iou:
                    best, best_iou = c, v
            if best is None:
                clusters.append([[it], list(it[3])])
            else:
                best[0].append(it)
                ws = sum(t[0] for t in best[0])
                best[1] = [sum(t[0] * t[3][k] for t in best[0]) / ws for k in range(4)]
        for members, box in clusters:
            in_cluster = {t[2] for t in members}
            w_sum = sum(weights.get(t[2], 1.0) for t in members)
            score = sum(t[0] for t in members) / w_sum if w_sum > 0 else 0.0
            n_models = sum(weights.get(m, 1.0) for m in in_cluster)
            score *= min(n_models, total_w) / total_w if total_w > 0 else 1.0
            sources = {}
            for t in members:
                sources[t[2]] = max(sources.get(t[2], 0.0), t[1])
            out.append({
                "label": label,
                "bbox": [round(v, 1) for v in box],
                "score": round(min(1.0, score), 4),
                "sources": {k: round(v, 4) for k, v in sorted(sources.items())},
                "agree": len(models) > 1 and all(m in in_cluster for m in models),
            })
    out.sort(key=lambda b: -b["score"])
    return out


def prompt_batches(phrases: list, count_tokens, limit: int) -> list:
    """Split `phrases` into batches whose prompt ("a. b. c.") has at most `limit` tokens by
    `count_tokens(text)`; a single phrase over the limit gets a batch of its own (the encoder
    truncates it). Order is kept."""
    batches, cur = [], []
    for p in phrases:
        trial = cur + [p]
        if cur and count_tokens(". ".join(trial) + ".") > limit:
            batches.append(cur)
            cur = [p]
        else:
            cur = trial
    if cur:
        batches.append(cur)
    return batches


def phrase_class(text: str, phrase_to_class: dict):
    """The class of a phrase MM-Grounding-DINO reported (it may return part of a prompt phrase or
    run two together): exact match, else the prompt phrase sharing the most words with it (ties:
    the longer phrase), else None."""
    t = " ".join(text.lower().replace(".", " ").split())
    if t in phrase_to_class:
        return phrase_to_class[t]
    words = set(t.split())
    best, best_n = None, 0
    for p, c in phrase_to_class.items():
        n = len(words & set(p.split()))
        if n > best_n or (n == best_n and n > 0 and best is not None and len(p) > len(best[0])):
            best, best_n = (p, c), n
    return best[1] if best else None


VLM_WORDS = ("person", "vehicle", "animal", "statue", "rock", "plant", "nothing")


def vlm_answer(text: str) -> str:
    """The first known word of a VLM answer ("A rock." -> "rock"; plurals folded), else the first
    word, lowercased."""
    words = [w.strip(".,!?;:\"'()[]").lower() for w in text.split()]
    words = [w for w in words if w]
    for w in words:
        s = w[:-1] if w.endswith("s") and w[:-1] in VLM_WORDS else w
        if s in VLM_WORDS or s in ("car", "truck", "dog", "cat", "people", "man", "woman", "tree", "bush"):
            return {"car": "vehicle", "truck": "vehicle", "dog": "animal", "cat": "animal", "people": "person",
                    "man": "person", "woman": "person", "tree": "plant", "bush": "plant"}.get(s, s)
    return words[0] if words else ""


def crop_box(b, w: int, h: int, pad: float = 0.2):
    """`b` padded by `pad` of its size on every side, clamped, as integer pixels."""
    bw, bh = b[2] - b[0], b[3] - b[1]
    x0 = max(0, int(math.floor(b[0] - pad * bw)))
    y0 = max(0, int(math.floor(b[1] - pad * bh)))
    x1 = min(w, int(math.ceil(b[2] + pad * bw)))
    y1 = min(h, int(math.ceil(b[3] + pad * bh)))
    return x0, y0, max(x1, x0 + 1), max(y1, y0 + 1)


def needs_vlm(b: dict) -> bool:
    """The VLM checks person boxes and boxes only one labeler found (of the person, vehicle and
    animal classes: its answers cover only those)."""
    c = category(b["label"])
    return c != "object" and (c == "person" or not b.get("agree", False))


def vlm_plan(b: dict, min_score: float):
    """Whether the VLM checks box `b`: "ask", "score" (it would, but the fused score is below
    `min_score`; 0 = no cutoff) or None (no check, see `needs_vlm`)."""
    if not needs_vlm(b):
        return None
    return "score" if min_score > 0 and b["score"] < min_score else "ask"


def answer_token_ids(tokenize) -> set:
    """The token ids that are a whole VLM answer word on their own (`VLM_WORDS`, lower case and
    capitalized; `tokenize(text)` -> ids without special tokens). The VLM stops after such a first
    token (`Vlm.ask`): it would only add the end-of-turn token, and `vlm_answer` of the word alone is
    the same. A word split into several tokens ("stat" + "ue") is not in the set and runs on."""
    ids = set()
    for w in VLM_WORDS:
        for v in (w, w.capitalize()):
            t = list(tokenize(v))
            if len(t) == 1:
                ids.add(int(t[0]))
    return ids


def first_token_done(new_tokens: list, ids: set) -> bool:
    """Whether a reply of `new_tokens` (ids generated so far) is complete by `answer_token_ids`:
    exactly one token, and it is an answer word."""
    return len(new_tokens) == 1 and int(new_tokens[0]) in ids


def chunks(items: list, n: int) -> list:
    """`items` in consecutive lists of at most `n` (n < 1 counts as 1), order kept."""
    n = max(1, int(n))
    return [items[i:i + n] for i in range(0, len(items), n)]


def self_test() -> int:
    n = 0

    def check(cond, what):
        nonlocal n
        if not cond:
            raise SystemExit(f"[autolabel] self-test failed: {what}")
        n += 1

    check({"person", "traffic light", "potted plant", "teddy bear", "dog", "hair drier"} <= COCO_NAMES, "coco names")
    check(len(COCO_NAMES) == 80, f"80 coco names, got {len(COCO_NAMES)}")
    check(coco_class("motorbike") == "motorcycle" and coco_class("raccoon") is None, "coco_class")
    check(abs(iou([0, 0, 2, 2], [1, 1, 3, 3]) - 1 / 7) < 1e-9, "iou")
    check(iou([0, 0, 1, 1], [2, 2, 3, 3]) == 0.0, "iou disjoint")
    kept = nms([
        {"label": "dog", "bbox": [0, 0, 10, 10], "score": 0.9},
        {"label": "dog", "bbox": [1, 0, 11, 10], "score": 0.8},
        {"label": "cat", "bbox": [1, 0, 11, 10], "score": 0.7},
        {"label": "dog", "bbox": [50, 50, 60, 60], "score": 0.3},
    ], 0.6)
    check([(k["label"], k["score"]) for k in kept] == [("dog", 0.9), ("cat", 0.7), ("dog", 0.3)], "nms")
    # WBF: two models agree on a person (fused box = score-weighted mean), one model alone on a car.
    fused = wbf({
        "gdino": [{"label": "person", "bbox": [10, 10, 50, 90], "score": 0.6},
                  {"label": "raccoon", "bbox": [200, 200, 240, 230], "score": 0.5}],
        "rfdetr": [{"label": "person", "bbox": [12, 10, 52, 90], "score": 0.9},
                   {"label": "car", "bbox": [100, 100, 180, 150], "score": 0.8}],
    }, {"gdino": 1.0, "rfdetr": 1.0}, 0.55, {"raccoon": ["gdino"]})
    by = {b["label"]: b for b in fused}
    p = by["person"]
    check(abs(p["score"] - 0.75) < 1e-9 and p["agree"], f"wbf agreed score {p}")
    check(abs(p["bbox"][0] - (10 * 0.6 + 12 * 0.9) / 1.5) < 0.051, f"wbf box {p['bbox']}")
    check(p["sources"] == {"gdino": 0.6, "rfdetr": 0.9}, "wbf sources")
    check(abs(by["car"]["score"] - 0.4) < 1e-9 and not by["car"]["agree"], "wbf single source halves")
    check(abs(by["raccoon"]["score"] - 0.5) < 1e-9 and not by["raccoon"]["agree"], "wbf one-model class")
    check([b["label"] for b in fused] == ["person", "raccoon", "car"], "wbf order")
    # Two boxes of one model in one cluster: the cluster still counts one model.
    two = wbf({"a": [{"label": "x", "bbox": [0, 0, 10, 10], "score": 0.8},
                     {"label": "x", "bbox": [0, 0, 10, 11], "score": 0.6}], "b": []},
              {"a": 1.0, "b": 1.0}, 0.55, {})
    check(len(two) == 1 and abs(two[0]["score"] - 0.35) < 1e-9, f"wbf same-model cluster {two}")
    # Prompt batches by a fake token count (one token per word plus one per period).
    count = lambda t: len(t.replace(".", " .").split())
    b = prompt_batches(["a", "b c", "d", "e f g"], count, 6)
    check(b == [["a", "b c"], ["d", "e f g"]], f"prompt batches {b}")
    check(prompt_batches(["x y z w v u t"], count, 3) == [["x y z w v u t"]], "oversized phrase")
    m = {"raccoon": "raccoon", "virginia opossum": "opossum", "white-tailed deer": "deer", "car": "car"}
    check(phrase_class("raccoon", m) == "raccoon", "phrase exact")
    check(phrase_class("opossum", m) == "opossum", "phrase partial")
    check(phrase_class("white-tailed deer raccoon", m) == "deer", "phrase run-together")
    check(phrase_class("tree", m) is None, "phrase unknown")
    check(vlm_answer("A rock.") == "rock" and vlm_answer("Plants") == "plant", "vlm answer")
    check(vlm_answer("person") == "person" and vlm_answer("It is a dog") == "animal", "vlm mapping")
    check(category("person") == "person" and category("truck") == "vehicle" and category("raccoon") == "animal", "category")
    check(category("package") == "object" and not needs_vlm({"label": "package", "agree": False}), "object class")
    check(crop_box([10, 10, 20, 30], 100, 100) == (8, 6, 22, 34), f"crop {crop_box([10, 10, 20, 30], 100, 100)}")
    check(crop_box([0, 0, 100, 100], 100, 100) == (0, 0, 100, 100), "crop clamp")
    check(needs_vlm({"label": "person", "agree": True}) and needs_vlm({"label": "dog", "agree": False}), "needs_vlm")
    check(not needs_vlm({"label": "dog", "agree": True}), "needs_vlm agreed animal")
    check(vlm_plan({"label": "person", "agree": True, "score": 0.1}, 0.0) == "ask", "vlm_plan no cutoff")
    check(vlm_plan({"label": "person", "agree": True, "score": 0.1}, 0.2) == "score", "vlm_plan below cutoff")
    check(vlm_plan({"label": "dog", "agree": False, "score": 0.2}, 0.2) == "ask", "vlm_plan at cutoff asks")
    check(vlm_plan({"label": "dog", "agree": True, "score": 0.05}, 0.2) is None, "vlm_plan no check wins")
    check(vlm_plan({"label": "package", "agree": False, "score": 0.9}, 0.2) is None, "vlm_plan object class")
    check(chunks([1, 2, 3, 4, 5], 2) == [[1, 2], [3, 4], [5]], "chunks")
    check(chunks([1, 2, 3], 8) == [[1, 2, 3]] and chunks([], 4) == [], "chunks short / empty")
    check(chunks([1, 2], 1) == [[1], [2]] and chunks([1, 2], 0) == [[1], [2]], "chunks of one")
    # Answer words as single tokens: a fake tokenizer that splits "statue" in two.
    vocab = {w: i for i, w in enumerate(["person", "Person", "vehicle", "animal", "rock", "plant", "nothing",
                                          "stat", "ue", "Stat", "Vehicle"])}
    fake_tok = lambda t: [vocab["stat" if t[0] == "s" else "Stat"], vocab["ue"]] if t.lower() == "statue" else \
        ([vocab[t]] if t in vocab else [100, 101])
    ids = answer_token_ids(fake_tok)
    check(ids == {vocab[w] for w in ("person", "Person", "vehicle", "Vehicle", "animal", "rock", "plant", "nothing")},
          f"answer_token_ids {sorted(ids)}")
    check(first_token_done([vocab["person"]], ids) and first_token_done([vocab["Vehicle"]], ids), "first token done")
    check(not first_token_done([vocab["stat"]], ids) and not first_token_done([], ids), "split word runs on")
    check(not first_token_done([vocab["person"], vocab["ue"]], ids), "only the first token stops")
    # vlm_check with fakes: batches of 2 of 3 crops, then a failing batch falls back to one at a time.
    class FakeImage:
        size = (100, 100)

        def crop(self, box):
            return box

    class FakeVlm:
        def __init__(self, fail):
            self.fail, self.calls = fail, []

        def ask(self, crop):
            self.calls.append(1)
            return "A person." if crop[0] < 50 else "rock"

        def ask_many(self, crops):
            if self.fail:
                raise RuntimeError("out of memory")
            self.calls.append(len(crops))
            return ["A person." if c[0] < 50 else "rock" for c in crops]

        def free(self):
            pass

    quiet, real_log = (lambda msg: None), globals()["log"]
    for fail, calls in ((False, [2, 1]), (True, [1, 1, 1])):
        fv, st = FakeVlm(fail), {}
        bs = [{"label": "person", "bbox": [10, 10, 20, 20]}, {"label": "dog", "bbox": [60, 60, 70, 70]},
              {"label": "person", "bbox": [5, 5, 9, 9]}]
        globals()["log"] = quiet
        try:
            vlm_check(fv, FakeImage(), bs, 2, "fake", st)
        finally:
            globals()["log"] = real_log
        check([(b["vlm"]["answer"], b["vlm"]["ok"]) for b in bs] == [("person", True), ("rock", False), ("person", True)],
              f"vlm_check answers (fail={fail}) {bs}")
        check(fv.calls == calls and st.get("fallbacks", 0) == (1 if fail else 0), f"vlm_check calls (fail={fail}) {fv.calls}")
    got = list(FrameReader([{"path": "/nonexistent/a.jpg"}, {"path": "/nonexistent/b.jpg"}], True,
                           opener=lambda path: path[-5]))
    check([(i, img, err) for i, _, img, err in got] == [(0, "a", None), (1, "b", None)], f"prefetch order {got}")

    def bad(path):
        if path.endswith("b"):
            raise OSError("broken")
        return path

    for pre in (False, True):
        got = [(i, img, str(err) if err else None) for i, _, img, err in
               FrameReader([{"path": "a"}, {"path": "b"}, {"path": "c"}], pre, opener=bad)]
        check(got == [(0, "a", None), (1, None, "broken"), (2, "c", None)], f"reader errors (prefetch={pre}) {got}")
    # Stopping early (an exception in the loop) ends the prefetch thread.
    r = FrameReader([{"path": str(k)} for k in range(10)], True, opener=lambda path: path)
    with r:
        for item in r:
            break
    check(r.thread is not None and not r.thread.is_alive(), "prefetch thread stops on early exit")
    event("self_test", passed=n)
    log(f"self-test: {n} checks passed")
    return 0


# ---------------------------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------------------------


def pick_device(want: str) -> str:
    import torch

    w = (want or "auto").lower()
    if w == "auto":
        if torch.cuda.is_available():
            return "cuda:0"
        if getattr(torch.backends, "mps", None) is not None and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if w == "cuda":
        return "cuda:0"
    return w


def watch_parent() -> None:
    """Exit when the service that started this process is gone (Unix; Windows: its job object)."""
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


def untie_decoder_heads(model, path: Path) -> int:
    """Load MM-Grounding-DINO's per-layer decoder heads as the checkpoint has them.

    The checkpoints carry one box head (and one class head) per decoder layer
    (`model.decoder.bbox_embed.<i>.*`), but transformers 5.17 declares them tied to layer 0's
    (`_tied_weights_keys`), so every refinement step used layer 0's head and the boxes came out
    a few pixels wide and the full frame high. Each layer gets its own copy with its own weights
    again. Returns the heads loaded (0 when the checkpoint shares them)."""
    import copy

    import torch
    from safetensors import safe_open

    f = safe_open(str(path / "model.safetensors"), "pt")
    keys = set(f.keys())
    loaded = 0
    for name in ("bbox_embed", "class_embed"):
        heads = getattr(model, name, None)
        if heads is None:
            continue
        new = torch.nn.ModuleList()
        for i, h in enumerate(heads):
            prefix = f"model.decoder.{name}.{i}." if i else f"{name}.0."
            sd = {k[len(prefix):]: f.get_tensor(k) for k in keys if k.startswith(prefix)}
            m = copy.deepcopy(h) if i else h
            if sd and i:
                missing, unexpected = m.load_state_dict(sd, strict=False)
                if unexpected:
                    fail(f"{path}: unexpected keys in {name}.{i}: {unexpected[:3]}")
                loaded += 1
            new.append(m)
        setattr(model, name, new)
        decoder = getattr(getattr(model, "model", None), "decoder", None)
        if decoder is not None:
            setattr(decoder, name, new)
    return loaded


class GroundingDino:
    """MM-Grounding-DINO through transformers' zero-shot detection API."""

    def __init__(self, path: Path, device: str, threshold: float, nms_iou: float, token_limit: int):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(path))
        model = AutoModelForZeroShotObjectDetection.from_pretrained(str(path))
        n = untie_decoder_heads(model, path)
        log(f"MM-Grounding-DINO: {n} per-layer decoder heads loaded")
        self.model = model.to(device).eval()
        self.device = device
        self.threshold = threshold
        self.nms_iou = nms_iou
        self.limit = token_limit

    def tokens(self, text: str) -> int:
        return len(self.processor.tokenizer(text)["input_ids"])

    def batches(self, phrases: list) -> list:
        return prompt_batches(phrases, self.tokens, self.limit)

    def detect(self, image, batches: list, phrase_to_class: dict) -> list:
        torch = self.torch
        w, h = image.size
        found = []
        for batch in batches:
            text = ". ".join(batch) + "."
            inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
            with torch.no_grad():
                outputs = self.model(**inputs)
            res = self.processor.post_process_grounded_object_detection(
                outputs, inputs.input_ids, threshold=self.threshold, text_threshold=self.threshold,
                target_sizes=[(h, w)],
            )[0]
            labels = res.get("text_labels", res.get("labels"))
            for box, score, lab in zip(res["boxes"].tolist(), res["scores"].tolist(), labels):
                cls = phrase_class(str(lab), phrase_to_class)
                if cls is None:
                    continue
                x0, y0, x1, y1 = (max(0.0, min(float(box[0]), w)), max(0.0, min(float(box[1]), h)),
                                  max(0.0, min(float(box[2]), w)), max(0.0, min(float(box[3]), h)))
                if x1 - x0 >= 1 and y1 - y0 >= 1:
                    found.append({"label": cls, "bbox": [x0, y0, x1, y1], "score": float(score)})
        return nms(found, self.nms_iou)


def native_grid_sample_on_mps() -> bool:
    """Make rfdetr's deformable attention use `F.grid_sample` on MPS while no gradient is needed.

    On MPS (and XLA) rfdetr samples through a gather-based bilinear path, written for training
    (grid_sample's backward has no MPS kernel); its forward has one, which gives the same values
    (measured: identical boxes and scores) and RF-DETR L about 10% faster. With grad enabled, or on
    another device, rfdetr's own function runs. Both are compared once on a random input first.
    Returns False when nothing was patched (no such function in this rfdetr, or the two differ)."""
    try:
        import rfdetr.models.ops.functions.ms_deform_attn_func as msda
    except ImportError:
        return False
    own = getattr(msda, "_bilinear_grid_sample", None)
    if own is None or getattr(own, "_autolabel_native", False):
        return own is not None
    import torch
    import torch.nn.functional as F

    try:
        g = torch.Generator().manual_seed(0)
        x = torch.randn(2, 8, 13, 17, generator=g).to("mps")
        grid = (torch.rand(2, 50, 4, 2, generator=g) * 2.4 - 1.2).to("mps")  # some points outside
        with torch.no_grad():
            diff = (own(x, grid, padding_mode="zeros", align_corners=False)
                    - F.grid_sample(x, grid, mode="bilinear", padding_mode="zeros", align_corners=False)).abs().max().item()
    except Exception as e:  # noqa: BLE001 - keep rfdetr's own path
        log(f"RF-DETR: grid_sample check failed ({type(e).__name__}: {e}); keeping rfdetr's sampling")
        return False
    if not diff <= 1e-4:
        log(f"RF-DETR: grid_sample differs from rfdetr's sampling by {diff}; keeping rfdetr's")
        return False

    def bilinear(input, grid, padding_mode="zeros", align_corners=False):
        if input.device.type == "mps" and not torch.is_grad_enabled():
            return F.grid_sample(input, grid, mode="bilinear", padding_mode=padding_mode, align_corners=align_corners)
        return own(input, grid, padding_mode=padding_mode, align_corners=align_corners)

    bilinear._autolabel_native = True
    msda._bilinear_grid_sample = bilinear
    return True


class RfDetr:
    """RF-DETR L (COCO) through the rfdetr package."""

    def __init__(self, weights: Path, device: str, threshold: float):
        import rfdetr
        from rfdetr.assets.coco_classes import COCO_CLASSES

        self.names = dict(COCO_CLASSES)
        self.model = rfdetr.RFDETRLarge(pretrain_weights=str(weights), device=device)
        self.threshold = threshold
        if str(device).startswith("mps") and native_grid_sample_on_mps():
            log("RF-DETR: deformable attention samples with grid_sample's MPS kernel")

    def detect(self, image, wanted: dict) -> list:
        det = self.model.predict(image, threshold=self.threshold, include_source_image=False)
        out = []
        for box, score, cid in zip(det.xyxy.tolist(), det.confidence.tolist(), det.class_id.tolist()):
            name = self.names.get(int(cid))
            cls = wanted.get(name) if name else None
            if cls is not None:
                out.append({"label": cls, "bbox": [float(v) for v in box], "score": float(score)})
        return out


class Vlm:
    """Qwen3-VL answering one word about a crop."""

    QUESTION = ("What is in this image: a person, a vehicle, an animal, a statue, a rock, a plant, or "
                "nothing? Answer with one word.")

    def __init__(self, path: Path, device: str, name: str, early_stop: bool = True):
        import torch
        from transformers import AutoModelForImageTextToText, AutoProcessor

        self.torch = torch
        dtype = torch.float16 if device.startswith(("mps", "cuda")) else torch.float32
        if device.startswith("cuda") and torch.cuda.is_bf16_supported():
            dtype = torch.bfloat16
        self.processor = AutoProcessor.from_pretrained(str(path))
        self.model = AutoModelForImageTextToText.from_pretrained(str(path), dtype=dtype).to(device).eval()
        self.device = device
        self.name = name
        tok = self.processor.tokenizer
        ids = answer_token_ids(lambda t: tok.encode(t, add_special_tokens=False)) if early_stop else set()
        self.answer_ids = torch.tensor(sorted(ids), device=device) if ids else None

    def stopping(self, prompt_len: int) -> dict:
        """`generate` kwargs that end a reply after its first token when that is an answer word
        (`first_token_done`; the next token would be end-of-turn): one decode step less per crop, same
        answer. Rows of a batch stop on their own."""
        if self.answer_ids is None:
            return {}
        from transformers import StoppingCriteria, StoppingCriteriaList

        torch, ids = self.torch, self.answer_ids

        class FirstWord(StoppingCriteria):
            def __call__(self, input_ids, scores, **kwargs):
                if input_ids.shape[1] != prompt_len + 1:
                    return torch.zeros(input_ids.shape[0], dtype=torch.bool, device=input_ids.device)
                return torch.isin(input_ids[:, -1], ids)

        return {"stopping_criteria": StoppingCriteriaList([FirstWord()])}

    def ask(self, crop) -> str:
        torch = self.torch
        crop = self.fit(crop)
        messages = [{"role": "user", "content": [{"type": "image", "image": crop}, {"type": "text", "text": self.QUESTION}]}]
        inputs = self.processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True,
                                                    return_dict=True, return_tensors="pt").to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                      **self.stopping(inputs["input_ids"].shape[1]))
        text = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
        return text.strip()

    @staticmethod
    def fit(crop):
        """The crop as `ask` sends it: at most 448 px on the long side, at least 32 on the short."""
        if max(crop.size) > 448:
            s = 448 / max(crop.size)
            crop = crop.resize((max(1, round(crop.size[0] * s)), max(1, round(crop.size[1] * s))))
        if min(crop.size) < 32:
            s = 32 / min(crop.size)
            crop = crop.resize((max(32, round(crop.size[0] * s)), max(32, round(crop.size[1] * s))))
        return crop

    def ask_many(self, crops: list) -> list:
        """`ask` for several crops in one `generate` call (left padding, the attention mask marks
        the padding); one answer per crop, in order."""
        if len(crops) == 1:
            return [self.ask(crops[0])]
        torch = self.torch
        conversations = [[{"role": "user", "content": [{"type": "image", "image": self.fit(c)},
                                                       {"type": "text", "text": self.QUESTION}]}] for c in crops]
        inputs = self.processor.apply_chat_template(
            conversations, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt",
            processor_kwargs={"padding": True, "padding_side": "left"},
        ).to(self.device)
        with torch.no_grad():
            out = self.model.generate(**inputs, max_new_tokens=5, do_sample=False,
                                      **self.stopping(inputs["input_ids"].shape[1]))
        texts = self.processor.batch_decode(out[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        return [t.strip() for t in texts]

    def free(self) -> None:
        """Return cached device memory (after a failed batch)."""
        torch = self.torch
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
        elif self.device == "mps":
            torch.mps.empty_cache()


def vlm_check(vlm: Vlm, image, boxes: list, batch: int, name: str, state: dict) -> None:
    """Sets `b["vlm"]` on every box of `boxes` (one frame) from the VLM's answer on its crop, `batch`
    crops per `generate` call. A failed batched call (e.g. out of memory) asks that frame's crops one
    at a time instead; the first failure of the run is logged, `state["fallbacks"]` counts them."""
    w, h = image.size
    crops = [image.crop(crop_box(b["bbox"], w, h)) for b in boxes]
    answers = []
    if batch > 1 and len(crops) > 1:
        try:
            for part in chunks(crops, batch):
                answers.extend(vlm.ask_many(part))
        except Exception as e:  # noqa: BLE001 - a batch that does not fit must not stop the run
            state["fallbacks"] = state.get("fallbacks", 0) + 1
            if state["fallbacks"] == 1:
                log(f"VLM batch of {batch} failed ({type(e).__name__}: {str(e)[:200]}); asking one crop at a "
                    "time for that frame (later failures are only counted)")
            vlm.free()
            answers = []
    if len(answers) != len(crops):
        answers = [vlm.ask(c) for c in crops]
    for b, raw in zip(boxes, answers):
        ans = vlm_answer(raw)
        exp = category(b["label"])
        b["vlm"] = {"model": name, "answer": ans, "expected": exp, "ok": ans == exp, "raw": raw[:40]}


class FrameReader:
    """The frames with their decoded RGB images, in order: `(index, frame, image, error)` (image None
    and the exception when it cannot be read). With `prefetch`, a thread decodes up to two frames
    ahead; leaving the `with` block (or exhausting it) stops the thread."""

    DEPTH = 2

    def __init__(self, frames: list, prefetch: bool, opener=None):
        self.frames = frames
        self.prefetch = prefetch
        self.opener = opener or self.open_rgb
        self.thread = None
        self.stop = threading.Event()
        self.q = queue.Queue(maxsize=self.DEPTH)

    @staticmethod
    def open_rgb(path):
        from PIL import Image

        return Image.open(path).convert("RGB")

    def load(self, i: int, fr: dict):
        try:
            return i, fr, self.opener(fr["path"]), None
        except Exception as e:  # noqa: BLE001 - one unreadable frame must not stop the run
            return i, fr, None, e

    def produce(self) -> None:
        for i, fr in enumerate(self.frames):
            item = self.load(i, fr)
            while not self.stop.is_set():
                try:
                    self.q.put(item, timeout=0.2)
                    break
                except queue.Full:
                    continue
            if self.stop.is_set():
                return
        while not self.stop.is_set():
            try:
                self.q.put(None, timeout=0.2)
                return
            except queue.Full:
                continue

    def __enter__(self):
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join()

    def __iter__(self):
        if not self.prefetch:
            for i, fr in enumerate(self.frames):
                yield self.load(i, fr)
            return
        self.thread = threading.Thread(target=self.produce, name="frame-prefetch", daemon=True)
        self.thread.start()
        try:
            while True:
                item = self.q.get()
                if item is None:
                    return
                yield item
        finally:
            self.close()


class Siglip:
    """SigLIP 2 zero-shot ranking of the animal classes on a crop."""

    DAY = ("a photo of a {}.", "a security camera photo of a {}.")
    NIGHT = ("an infrared night-vision photo of a {}.", "a grainy black and white security camera photo of a {} at night.")

    def __init__(self, path: Path, device: str):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(str(path))
        self.model = AutoModel.from_pretrained(str(path)).to(device).eval()
        self.device = device
        self.cache = {}

    def text_features(self, classes: tuple, night: bool, synonyms: dict):
        key = (classes, night)
        if key in self.cache:
            return self.cache[key]
        torch = self.torch
        feats = []
        for c in classes:
            names = synonyms.get(c) or [c]
            prompts = [t.format(n) for n in names for t in (self.NIGHT if night else self.DAY)]
            inputs = self.processor(text=prompts, padding="max_length", max_length=64, return_tensors="pt").to(self.device)
            with torch.no_grad():
                f = self.model.get_text_features(**inputs)
            f = getattr(f, "pooler_output", f)
            f = torch.nn.functional.normalize(f, dim=-1).mean(0)
            feats.append(torch.nn.functional.normalize(f, dim=-1))
        t = torch.stack(feats)
        self.cache[key] = t
        return t

    def rank(self, crop, classes: tuple, night: bool, synonyms: dict) -> dict:
        torch = self.torch
        t = self.text_features(classes, night, synonyms)
        inputs = self.processor(images=crop, return_tensors="pt").to(self.device)
        with torch.no_grad():
            f = self.model.get_image_features(**inputs)
        f = getattr(f, "pooler_output", f)
        f = torch.nn.functional.normalize(f, dim=-1)
        logits = (f @ t.T) * self.model.logit_scale.exp() + self.model.logit_bias
        p = torch.softmax(logits[0], dim=-1).tolist()
        return {c: round(float(v), 4) for c, v in zip(classes, p)}


# ---------------------------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------------------------


def self_check() -> int:
    import importlib

    for mod in ("numpy", "PIL", "torch", "transformers", "rfdetr"):
        m = importlib.import_module(mod)
        log(f"self-check: {mod} {getattr(m, '__version__', '?')}")
    import torch

    info = {"torch": torch.__version__, "cuda": torch.cuda.is_available(),
            "mps": bool(getattr(torch.backends, "mps", None) and torch.backends.mps.is_available())}
    event("self_check", **info)
    self_test()
    return 0


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--self-check", action="store_true")
    p.add_argument("--frames", type=Path)
    p.add_argument("--out", type=Path)
    p.add_argument("--classes", help="JSON list of the class names")
    p.add_argument("--synonyms", type=Path, help="JSON file: {class: [prompt phrases]}")
    p.add_argument("--categories", type=Path, help="JSON file: {class: person|vehicle|animal|object}")
    p.add_argument("--gdino", type=Path, help="MM-Grounding-DINO checkpoint directory")
    p.add_argument("--rfdetr", type=Path, help="RF-DETR L checkpoint (.pth)")
    p.add_argument("--vlm", type=Path, help="Qwen3-VL checkpoint directory (optional)")
    p.add_argument("--vlm-name", default="qwen3-vl")
    p.add_argument("--siglip", type=Path, help="SigLIP 2 checkpoint directory (optional)")
    p.add_argument("--device", default="auto")
    p.add_argument("--gdino-threshold", type=float, default=0.2)
    p.add_argument("--rfdetr-threshold", type=float, default=0.3)
    p.add_argument("--nms-iou", type=float, default=0.6)
    p.add_argument("--wbf-iou", type=float, default=0.55)
    p.add_argument("--token-limit", type=int, default=200, help="most tokens of one MM-Grounding-DINO prompt (its encoder takes 256)")
    p.add_argument("--vlm-batch", type=int, default=1, help="crops of one frame per VLM generate call (1 = one at a time)")
    p.add_argument("--vlm-min-score", type=float, default=0.0,
                   help="skip the VLM on boxes whose fused score is below this (0 = check every box needs_vlm picks)")
    p.add_argument("--prefetch", action="store_true", help="decode the next frames on a background thread")
    a = p.parse_args()
    if a.self_test or a.self_check:
        return a
    if a.vlm_batch < 1:
        p.error("--vlm-batch must be at least 1")
    if not 0.0 <= a.vlm_min_score <= 1.0:
        p.error("--vlm-min-score must be between 0 and 1")
    missing = [k for k in ("frames", "out", "classes") if getattr(a, k) is None]
    if missing:
        p.error("missing: " + ", ".join("--" + m for m in missing))
    if a.gdino is None and a.rfdetr is None:
        p.error("give --gdino and/or --rfdetr")
    return a


def main() -> int:
    a = parse_args()
    # The libraries' deprecation notices (transformers, torch.jit) would fill the job's log tail
    # on the Training page instead of the progress lines.
    warnings.filterwarnings("ignore", category=FutureWarning)
    if a.self_test:
        return self_test()
    if a.self_check:
        return self_check()
    watch_parent()

    classes = [c.strip() for c in json.loads(a.classes) if c.strip()]
    synonyms = dict(DEFAULT_SYNONYMS)
    if a.synonyms:
        synonyms.update({k: v for k, v in json.loads(a.synonyms.read_text(encoding="utf-8")).items() if v})
    if a.categories:
        CATEGORIES.update(json.loads(a.categories.read_text(encoding="utf-8")))
    frames = [json.loads(l) for l in a.frames.read_text(encoding="utf-8").splitlines() if l.strip()]
    # torch reads this once, when it is imported (pick_device imports it): set later it does nothing. It
    # only affects the MPS backend. No op of the four labelers needs it with torch 2.14 (measured).
    if sys.platform == "darwin":
        os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    device = pick_device(a.device)
    event("start", frames=len(frames), classes=classes, device=device)

    phrase_to_class = {}
    for c in classes:
        for ph in synonyms.get(c) or [c]:
            phrase_to_class[" ".join(ph.lower().split())] = c
    coco_wanted = {}
    for c in classes:
        n = coco_class(c)
        if n is not None:
            coco_wanted.setdefault(n, c)
    labelers = {}
    t0 = time.time()
    if a.gdino:
        labelers["gdino"] = GroundingDino(a.gdino, device, a.gdino_threshold, a.nms_iou, a.token_limit)
        batches = labelers["gdino"].batches(list(phrase_to_class))
        event("model", name="gdino", seconds=round(time.time() - t0, 1), prompts=len(batches))
        log(f"MM-Grounding-DINO prompts: {[' . '.join(b) for b in batches]}")
    if a.rfdetr and coco_wanted:
        t = time.time()
        labelers["rfdetr"] = RfDetr(a.rfdetr, device, a.rfdetr_threshold)
        event("model", name="rfdetr", seconds=round(time.time() - t, 1))
    vlm = siglip = None
    if a.vlm:
        t = time.time()
        vlm = Vlm(a.vlm, device, a.vlm_name)
        event("model", name="vlm", seconds=round(time.time() - t, 1))
    if a.siglip:
        t = time.time()
        siglip = Siglip(a.siglip, device)
        event("model", name="siglip", seconds=round(time.time() - t, 1))
    participants = {}
    for c in classes:
        participants[c] = sorted(m for m in labelers if m == "gdino" or (m == "rfdetr" and coco_class(c) is not None))
    animals = tuple(c for c in classes if category(c) == "animal")
    weights = {m: 1.0 for m in labelers}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    totals = {"boxes": 0, "agreed": 0, "vlm": 0}
    vlm_state = {}
    if vlm is not None and (a.vlm_batch > 1 or a.vlm_min_score > 0):
        log(f"VLM: {a.vlm_batch} crop(s) per call, skipped below fused score {a.vlm_min_score}")
    reader = FrameReader(frames, a.prefetch)
    with open(a.out, "w", encoding="utf-8") as out, reader:
        for i, fr, image, e in reader:
            ms = {}
            if image is None:
                log(f"{fr.get('frame')}: cannot read the image ({e}); skipped")
                event("frame", index=i + 1, frames=len(frames), frame=fr.get("frame"), boxes=0, error=str(e))
                continue
            w, h = image.size
            per = {}
            if "gdino" in labelers:
                t = time.time()
                per["gdino"] = labelers["gdino"].detect(image, batches, phrase_to_class)
                ms["gdino"] = round((time.time() - t) * 1000)
            if "rfdetr" in labelers:
                t = time.time()
                per["rfdetr"] = labelers["rfdetr"].detect(image, coco_wanted)
                ms["rfdetr"] = round((time.time() - t) * 1000)
            fused = wbf(per, weights, a.wbf_iou, participants)
            night = fr.get("scene") in ("night", "ir")
            if vlm is not None:
                ask = []
                for b in fused:
                    plan = vlm_plan(b, a.vlm_min_score)
                    if plan == "ask":
                        ask.append(b)
                    elif plan == "score":
                        b["vlm_skipped"] = "score"
                if ask:
                    t = time.time()
                    vlm_check(vlm, image, ask, a.vlm_batch, a.vlm_name, vlm_state)
                    ms["vlm"] = round((time.time() - t) * 1000)
                    totals["vlm"] += len(ask)
            for b in fused:
                if siglip is not None and b["label"] in animals and len(animals) > 1:
                    t = time.time()
                    crop = image.crop(crop_box(b["bbox"], w, h, 0.1))
                    b["species_scores"] = siglip.rank(crop, animals, night, synonyms)
                    ms["siglip"] = ms.get("siglip", 0) + round((time.time() - t) * 1000)
            totals["boxes"] += len(fused)
            totals["agreed"] += sum(1 for b in fused if b["agree"])
            out.write(json.dumps({"frame": fr["frame"], "width": w, "height": h, "boxes": fused, "ms": ms},
                                 separators=(",", ":")) + "\n")
            out.flush()
            event("frame", index=i + 1, frames=len(frames), frame=fr["frame"], boxes=len(fused), ms=ms)
    if vlm_state.get("fallbacks"):
        log(f"VLM: {vlm_state['fallbacks']} frame(s) fell back to one crop at a time")
    event("done", frames=len(frames), seconds=round(time.time() - t0, 1), device=device, **totals)
    return 0


if __name__ == "__main__":
    sys.exit(main())
