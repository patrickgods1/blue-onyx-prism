#!/usr/bin/env python
"""Measure auto-label proposals against known boxes (docs/TRAINING.md, "Measuring auto-label").

Answers two questions with the same numbers on every run: how many proposals are real (precision),
and how many real objects a filter would hide from the reviewer (recall lost). Nothing here changes
a label file; everything it writes goes under the folder named by --out.

    # 1. Known boxes, from a built-in benchmark set or a finished training run's dataset
    python scripts/research/eval_autolabel.py truth --bench assets/bench/coco-cctv.json \\
        --images <data root>/bench/coco-cctv --out e/coco
    python scripts/research/eval_autolabel.py truth --run <training dir>/<run id>/data --out e/back

    # 2. Proposals: scripts/autolabel.py --frames e/coco/frames.jsonl --out e/coco/proposals.jsonl ...

    # 3. The one-line VLM check on each proposal (only needed for proposals made before it shipped)
    python scripts/research/eval_autolabel.py yesno --dir e/coco --vlm <qwen3-vl-2b dir> --device mps

    # 4. The report
    python scripts/research/eval_autolabel.py score --dir e/coco [--dir e/back ...] [--json report.json]

Matching: a proposal is real when it has the label of an unmatched known box at IoU >= 0.5 (highest
score first). Recall lost is per known object: found = some proposal of its label overlaps it at
IoU >= 0.5; hidden = no such proposal survives the filter. Known boxes of classes outside the run
are ignored, and proposals of classes a set does not annotate are left out of its precision (a car
in a dog-only set is unknown, not false).

    python scripts/research/eval_autolabel.py --self-test
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

MATCH_IOU = 0.5
# Below this fused score a proposal is hidden (docs/TRAINING.md, "Measuring auto-label").
HIDE_BELOW = 0.15
# At or above this, with a VLM yes, a proposal is likely right.
LIKELY_SCORE = 0.25
# The public sets' spellings of the service's classes.
ALIASES = {"motorbike": "motorcycle", "aeroplane": "airplane", "pottedplant": "potted plant",
           "tvmonitor": "tv", "sofa": "couch", "diningtable": "dining table"}
# A class fails the gate when the filter hides more than this share of the real objects found.
GATE_HIDDEN = 0.05
# Fewer real objects than this: too few to pass or fail a class.
GATE_MIN_OBJECTS = 20


def iou(a, b) -> float:
    x0, y0, x1, y1 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, x1 - x0) * max(0.0, y1 - y0)
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


def yolo_box(line: str, names: list, w: int, h: int):
    """One line of a YOLO label file as `(label, [x0, y0, x1, y1])` in pixels."""
    c, x, y, bw, bh = line.split()[:5]
    x, y, bw, bh = float(x), float(y), float(bw), float(bh)
    return names[int(c)], [(x - bw / 2) * w, (y - bh / 2) * h, (x + bw / 2) * w, (y + bh / 2) * h]


def bench_box(obj: dict):
    """One object of a benchmark manifest as `(label, [x0, y0, x1, y1])`."""
    box = obj.get("bbox") or obj.get("box")
    label = obj["label"]
    return ALIASES.get(label, label), [float(v) for v in box]


def keep(b: dict) -> bool:
    """The filter under test: what the reviewer sees. A VLM no never hides a box: on public sets
    40% of the boxes it said no to were real."""
    return b["score"] >= HIDE_BELOW


def answer(b: dict):
    """The VLM's yes or no: a replay's `yesno`, else what the labeling run stored on the box."""
    return b.get("yesno") or (b.get("vlm") or {}).get("answer")


def tier(b: dict) -> str:
    if b["score"] < HIDE_BELOW:
        return "hidden"
    return "likely" if b["score"] >= LIKELY_SCORE and answer(b) == "yes" else "unlikely"


def score_frames(frames: list, filters=keep) -> dict:
    """Totals of `frames` (`{"truth": [(label, box)], "boxes": [proposal], "known": classes|None}`)."""
    t = dict(frames=0, proposals=0, real=0, kept=0, kept_real=0)
    tiers = defaultdict(lambda: dict(boxes=0, real=0))
    per = defaultdict(lambda: dict(objects=0, found=0, hidden=0, proposals=0, real=0, kept=0, kept_real=0))
    for f in frames:
        t["frames"] += 1
        known = f.get("known")
        unmatched = list(enumerate(f["truth"]))
        for label, _ in f["truth"]:
            per[label]["objects"] += 1
        for b in sorted(f["boxes"], key=lambda x: -x["score"]):
            if known is not None and b["label"] not in known:
                continue
            best, at = 0.0, None
            for i, (idx, (label, box)) in enumerate(unmatched):
                if label == b["label"]:
                    v = iou(b["bbox"], box)
                    if v > best:
                        best, at = v, i
            real = best >= MATCH_IOU
            if real:
                unmatched.pop(at)
            kept = filters(b)
            tiers[tier(b)]["boxes"] += 1
            tiers[tier(b)]["real"] += real
            c = per[b["label"]]
            for d in (t, c):
                d["proposals"] += 1
                d["real"] += real
                d["kept"] += kept
                d["kept_real"] += real and kept
        for label, box in f["truth"]:
            near = [b for b in f["boxes"] if b["label"] == label and iou(b["bbox"], box) >= MATCH_IOU]
            if near:
                per[label]["found"] += 1
                per[label]["hidden"] += not any(filters(b) for b in near)
    t["classes"] = {k: dict(v) for k, v in sorted(per.items())}
    t["tiers"] = {k: dict(v) for k, v in sorted(tiers.items())}
    return t


def gate(classes: dict) -> dict:
    """pass / fail / unknown per class, from the filter's share of hidden real objects."""
    out = {}
    for name, c in classes.items():
        if c["found"] < GATE_MIN_OBJECTS:
            out[name] = "unknown"
        else:
            out[name] = "fail" if c["hidden"] / c["found"] > GATE_HIDDEN else "pass"
    return out


def share(a: int, b: int) -> str:
    return f"{a / b:6.1%}" if b else "     -"


def report(name: str, t: dict) -> None:
    fr = max(t["frames"], 1)
    print(f"\n{name}: {t['frames']} frames")
    print(f"  now      {t['proposals'] / fr:5.2f} boxes/frame, {share(t['real'], t['proposals'])} real")
    print(f"  shown    {t['kept'] / fr:5.2f} boxes/frame, {share(t['kept_real'], t['kept'])} real")
    for k, v in t["tiers"].items():
        print(f"  {k:<8} {v['boxes'] / fr:5.2f} boxes/frame, {share(v['real'], v['boxes'])} real")
    print(f"  {'class':<12}{'objects':>8}{'found':>7}{'hidden':>8}{'lost':>8}{'proposals':>11}{'real':>7}{'kept':>6}")
    verdict = gate(t["classes"])
    for k, c in t["classes"].items():
        print(f"  {k:<12}{c['objects']:>8}{c['found']:>7}{c['hidden']:>8}{share(c['hidden'], c['found']):>8}"
              f"{c['proposals']:>11}{c['real']:>7}{c['kept']:>6}  {verdict[k]}")


def read_jsonl(path: Path) -> list:
    if not path.exists():
        return []
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def write_truth(out: Path, rows: list) -> None:
    out.mkdir(parents=True, exist_ok=True)
    with (out / "truth.jsonl").open("w") as t, (out / "frames.jsonl").open("w") as f:
        for r in rows:
            t.write(json.dumps(r) + "\n")
            f.write(json.dumps({k: r[k] for k in ("frame", "path", "width", "height", "scene")}) + "\n")
    print(f"{len(rows)} frames in {out}")


def truth_from_bench(a) -> list:
    m = json.loads(Path(a.bench).read_text())
    rows = []
    scored = m.get("scored_labels")
    known = sorted({ALIASES.get(x, x) for x in scored}) if scored else None
    for i in m["images"]:
        p = Path(a.images) / i["file"]
        if not p.is_file() or i.get("objects") is None:
            continue
        tags = i.get("tags") or []
        scene = "ir" if "ir" in tags else "night" if "night" in tags else "day"
        boxes = [bench_box(o) for o in i["objects"] if (o.get("bbox") or o.get("box")) and not o.get("ignore")]
        rows.append({"frame": i["file"], "path": str(p), "width": i["width"], "height": i["height"], "scene": scene,
                     "truth": boxes, "known": known})
    return rows


def truth_from_run(a) -> list:
    from PIL import Image

    data = Path(a.run)
    names = [x.strip() for x in (data / "classes.txt").read_text().splitlines() if x.strip()]
    rows = []
    for sub in ("", "heldout"):
        root = data / sub if sub else data
        for p in sorted((root / "images").glob("*.jpg")):
            if p.name.startswith(".") or "._" in p.name:
                continue
            w, h = Image.open(p).size
            lab = root / "labels" / (p.stem + ".txt")
            lines = [x for x in lab.read_text().splitlines() if x.strip()] if lab.exists() else []
            scene = "ir" if ".ir." in p.name or p.name.startswith("s0-ir.") else "day"
            rows.append({"frame": f"{sub or 'train'}/{p.name}", "path": str(p), "width": w, "height": h, "scene": scene,
                         "truth": [yolo_box(x, names, w, h) for x in lines], "known": None})
    return rows


def ask_yesno(a) -> int:
    """Ask `autolabel.vlm_question` about the proposals of --dir, resuming a stopped run."""
    import importlib.util

    from PIL import Image

    spec = importlib.util.spec_from_file_location("autolabel", Path(__file__).parent.parent / "autolabel.py")
    al = importlib.util.module_from_spec(spec)
    sys.modules["autolabel"] = al
    spec.loader.exec_module(al)
    d = Path(a.dir)
    paths = {r["frame"]: r["path"] for r in read_jsonl(d / "truth.jsonl")}
    done = {(r["frame"], r["i"]) for r in read_jsonl(d / "yesno.jsonl")}
    vlm = al.Vlm(Path(a.vlm), a.device, "qwen3-vl-2b", early_stop=False)
    with (d / "yesno.jsonl").open("a") as out:
        for fr in read_jsonl(d / "proposals.jsonl"):
            if not paths.get(fr["frame"]):
                continue
            image = None
            for i, b in enumerate(fr["boxes"]):
                if (fr["frame"], i) in done or b["score"] < a.min_score:
                    continue
                image = image or Image.open(paths[fr["frame"]]).convert("RGB")
                raw = vlm.ask(image.crop(al.crop_box(b["bbox"], image.width, image.height)), al.vlm_question(b["label"]))
                out.write(json.dumps({"frame": fr["frame"], "i": i, "answer": raw.strip().lower()[:6]}) + "\n")
                out.flush()
    return 0


def load(d: Path) -> list:
    truth = {r["frame"]: r for r in read_jsonl(d / "truth.jsonl")}
    answers = defaultdict(dict)
    for r in read_jsonl(d / "yesno.jsonl"):
        answers[r["frame"]][r["i"]] = "yes" if r["answer"].startswith("yes") else "no"
    out = []
    for fr in read_jsonl(d / "proposals.jsonl"):
        t = truth.get(fr["frame"])
        if t is None:
            continue
        boxes = fr["boxes"]
        for i, b in enumerate(boxes):
            if i in answers[fr["frame"]]:
                b["yesno"] = answers[fr["frame"]][i]
        out.append({"truth": [(x[0], x[1]) for x in t["truth"]], "boxes": boxes, "known": t.get("known"),
                    "scene": t["scene"]})
    return out


def score(a) -> int:
    every, doc = [], {}
    for d in a.dir:
        frames = load(Path(d))
        every.extend(frames)
        doc[Path(d).name] = score_frames(frames)
        report(Path(d).name, doc[Path(d).name])
        for scene in ("night", "ir"):
            part = [f for f in frames if f["scene"] == scene]
            if part and len(part) < len(frames):
                report(f"{Path(d).name} ({scene})", score_frames(part))
    if len(a.dir) > 1:
        doc["all"] = score_frames(every)
        report("all", doc["all"])
    best = doc.get("all") or next(iter(doc.values()))
    print("\nGate:", json.dumps(gate(best["classes"])))
    if a.json:
        Path(a.json).write_text(json.dumps(doc, indent=1))
    return 0


def self_test() -> int:
    bad = 0

    def check(cond, what):
        nonlocal bad
        if not cond:
            bad += 1
            print(f"FAIL: {what}")

    box = [10.0, 10.0, 20.0, 20.0]
    check(abs(iou(box, box) - 1) < 1e-9 and iou(box, [30, 30, 40, 40]) == 0, "iou")
    check(yolo_box("1 0.5 0.5 0.5 0.5", ["a", "b"], 100, 200) == ("b", [25.0, 50.0, 75.0, 150.0]), "yolo box")
    check(bench_box({"label": "motorbike", "bbox": [1, 2, 3, 4]}) == ("motorcycle", [1.0, 2.0, 3.0, 4.0]), "bench box")
    frames = [{"truth": [("car", box)], "boxes": [
        {"label": "car", "bbox": box, "score": 0.9, "yesno": "yes"},
        {"label": "dog", "bbox": box, "score": 0.9, "yesno": "no"},
        {"label": "cat", "bbox": [0, 0, 5, 5], "score": 0.1},
    ], "known": None}]
    t = score_frames(frames)
    check(t["proposals"] == 3 and t["real"] == 1 and t["kept"] == 2 and t["kept_real"] == 1, f"totals {t}")
    check(t["classes"]["car"]["found"] == 1 and t["classes"]["car"]["hidden"] == 0, "car found, kept")
    check(t["tiers"]["likely"]["real"] == 1 and t["tiers"]["unlikely"]["boxes"] == 1, f"tiers {t['tiers']}")
    check(t["tiers"]["hidden"]["boxes"] == 1, "a weak box is hidden")
    # A no ranks a box lower and never hides it; a weak score hides the object.
    frames[0]["boxes"][0]["yesno"] = "no"
    t = score_frames(frames)
    check(t["classes"]["car"]["hidden"] == 0 and t["tiers"].get("likely") is None, "a no does not hide")
    frames[0]["boxes"][0]["score"] = 0.1
    check(score_frames(frames)["classes"]["car"]["hidden"] == 1, "a real object can be hidden")
    frames[0]["boxes"][0]["score"] = 0.9
    check(tier({"score": 0.9, "vlm": {"answer": "yes"}}) == "likely", "the run's own answer counts")
    # A class the set does not annotate is left out of precision.
    frames[0]["known"] = ["car"]
    check(score_frames(frames)["proposals"] == 1, "unknown classes are unscored")
    many = {"a": dict(found=100, hidden=4), "b": dict(found=100, hidden=6), "c": dict(found=5, hidden=5)}
    check(gate(many) == {"a": "pass", "b": "fail", "c": "unknown"}, "gate")
    print(f"self-test: {'ok' if not bad else f'{bad} failed'}")
    return 1 if bad else 0


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--self-test", action="store_true")
    sub = p.add_subparsers(dest="cmd")
    t = sub.add_parser("truth")
    t.add_argument("--bench")
    t.add_argument("--images")
    t.add_argument("--run")
    t.add_argument("--out", required=True)
    y = sub.add_parser("yesno")
    y.add_argument("--dir", required=True)
    y.add_argument("--vlm", required=True)
    y.add_argument("--device", default="mps")
    y.add_argument("--min-score", type=float, default=HIDE_BELOW)
    s = sub.add_parser("score")
    s.add_argument("--dir", action="append", required=True)
    s.add_argument("--json")
    a = p.parse_args()
    if a.self_test:
        return self_test()
    if a.cmd == "truth":
        rows = truth_from_run(a) if a.run else truth_from_bench(a)
        write_truth(Path(a.out), rows)
        return 0
    if a.cmd == "yesno":
        return ask_yesno(a)
    if a.cmd == "score":
        return score(a)
    p.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
