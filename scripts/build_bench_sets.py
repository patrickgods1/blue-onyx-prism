#!/usr/bin/env python3
"""Build the pinned benchmark image manifests in assets/bench/.

Sets curated from public datasets (images are NOT committed; the manifests pin URL + SHA-256 +
size and the ground truth, and the benchmark downloads on demand):

  bmd45-cctv    real Bengaluru Safe City CCTV frames (BMD-45 val, CC BY 4.0), vehicles only
  exdark-night  low-light / night photos (ExDark test split via dronefreak/ExDark, BSD-3-Clause)
  exdark-ir     exdark-night converted to grayscale at load time ("transform": "grayscale",
                "images_from": "exdark-night"): a rough NIR proxy, no images of its own
  coco-cctv     COCO val2017 subset with CCTV-relevant content (annotations CC BY 4.0)
  nir-night     real night IR (monochrome) camera-trap frames from LILA BC (Caltech Camera
                Traps, ENA24, NACTI; CDLA-Permissive-1.0): vehicles, dogs, cats, raccoons and
                hard negatives (empty frames)

Grayscale (IR) test, used for the `grayscale` tag and the nir-night selection (keep in sync with
the service's image_stats): on a 64x64 grid of pixels (x = floor((i + 0.5) * w / 64), same for
y; every pixel when the image is smaller), chroma spread = mean((|R-G| + |G-B|) / 2) on 0..255,
and tint spread = the mean absolute residual of R-G and of G-B after a least-squares fit
d = a + b * luma (BT.601), averaged (an offset or gain tint, e.g. a green or purple IR frame).
Monochrome when either is below 3.0. Same rule and constants as src/image_stats.rs.

Selection is deterministic for a given --seed and pinned source revision: candidates are
sorted, shuffled with a seeded RNG and picked greedily per complexity bucket, favouring
under-represented labels and tags. Each selected image is downloaded into the work dir,
decoded with Pillow, and its sha256/size/width/height are written to the manifest.

Usage (Python 3.10+, needs Pillow; see scripts/requirements-bench.txt):

  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src            # all sets
  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src --set coco-cctv
  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src --validate  # re-check
  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src --retag-grayscale coco-cctv
  .venv/bin/python -I scripts/build_bench_sets.py --work /tmp/bench-src --set nir-night \
      --review-dir /tmp/nir-night-review [--exclude excluded.txt] [--nacti-persons]

nir-night picks a seeded, deterministic set; candidate negatives (LILA "empty" frames can hold
unlabelled animals or people) are written to a review contact sheet (--review-dir). Put the file
names to drop, one per line, in an --exclude file and re-run: the next candidates replace them.

Downloads are treated as untrusted data: file names are sanitised, archives are read by
member name only (nothing is extracted to paths from the archive), nothing is executed.
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import random
import re
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
import zipfile
from collections import Counter
from pathlib import Path

try:
    from PIL import Image, ImageStat
except ImportError:  # pragma: no cover
    sys.exit("Pillow is required: pip install -r scripts/requirements-bench.txt")

REPO = Path(__file__).resolve().parent.parent
OUT_DIR = REPO / "assets" / "bench"
COCO80_YAML = REPO / "assets" / "coco_classes.yaml"
USER_AGENT = "blue-onyx-prism-bench-builder/1.0"

BMD45_REPO = "iisc-aim/BMD-45"
BMD45_REV = "3e3f6a0b4834d282890c34986216c487914c5dff"
EXDARK_REPO = "dronefreak/ExDark"
EXDARK_REV = "980c106507f08ce9200b84b8308028eea1f41c97"
# images.cocodataset.org is an S3 bucket whose https certificate does not match the host name,
# so the https path-style S3 URL is used (same objects, verified byte-identical).
COCO_BASE = "https://s3.amazonaws.com/images.cocodataset.org"
COCO_ANN_ZIP = f"{COCO_BASE}/annotations/annotations_trainval2017.zip"
COCO_ANN_ZIP_SHA256 = "113a836d90195ee1f884e704da6304dfaaecff1f023f49b6ca93c4aaae470268"

SMALL_AREA = 32 * 32
NIGHT_LUMA = 50.0  # mean Pillow "L" (0..255) below this => night / low-light
GRAY_SPREAD = 3.0  # chroma spread (0..255) below this => monochrome / IR (see the module docs)
GRAY_GRID = 64
GRAY_RULE = "v2-fit"  # bump when the rule changes: nir-night's pixel-check cache is keyed by it
EMPTY_FRACTION = 0.07
CCTV_FOCUS = ["person", "bicycle", "car", "motorbike", "bus", "truck", "bird", "cat", "dog", "horse"]

DEFAULTS = {
    # BMD-45 frames are 1920x1080 PNG, ~3.3 MB each: the 150 MB budget caps this set (~45 images).
    "bmd45-cctv": {"count": 160, "max_mb": 150.0},
    "exdark-night": {"count": 160, "max_mb": 150.0},
    "coco-cctv": {"count": 180, "max_mb": 150.0},
    "exdark-ir": {"count": 0, "max_mb": 0.0},
    "nir-night": {"count": 220, "max_mb": 150.0},
}


# --------------------------------------------------------------------------------------------
# helpers


def log(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def load_coco80() -> list[str]:
    names = []
    for line in COCO80_YAML.read_text(encoding="utf-8").splitlines():
        m = re.match(r"^\s*-\s+(.+?)\s*$", line)
        if m:
            names.append(m.group(1))
    if len(names) != 80:
        raise SystemExit(f"expected 80 names in {COCO80_YAML}, got {len(names)}")
    return names


def http_get(url: str, *, retries: int = 4, timeout: float = 60.0) -> tuple[bytes, dict]:
    last: Exception | None = None
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                return resp.read(), dict(resp.headers.items())
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (403, 404):
                raise
            last = e
            time.sleep(1.5 * (attempt + 1))
    raise RuntimeError(f"GET {url} failed: {last}")


def http_head(url: str) -> tuple[int, int | None]:
    req = urllib.request.Request(url, method="HEAD", headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(req, timeout=60) as resp:
        size = resp.headers.get("Content-Length")
        return resp.status, int(size) if size else None


def safe_name(name: str) -> str:
    base = name.replace("\\", "/").split("/")[-1]
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,200}", base) or ".." in base:
        raise ValueError(f"unsafe file name from source: {name!r}")
    return base


def cached(path: Path, url: str, expect_sha: str | None = None) -> bytes:
    if path.exists():
        data = path.read_bytes()
        if expect_sha is None or hashlib.sha256(data).hexdigest() == expect_sha:
            return data
    log(f"  GET {url}")
    data, _ = http_get(url, timeout=600)
    if expect_sha and hashlib.sha256(data).hexdigest() != expect_sha:
        raise RuntimeError(f"sha256 mismatch for {url}")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".part")
    tmp.write_bytes(data)
    tmp.replace(path)
    return data


def hf_resolve(repo: str, rev: str, path: str) -> str:
    return f"https://huggingface.co/datasets/{repo}/resolve/{rev}/{urllib.parse.quote(path)}"


def hf_tree(repo: str, rev: str, path: str, cache: Path) -> list[dict]:
    """List a dataset folder at a pinned revision (follows Link: rel=next pagination)."""
    if cache.exists():
        return json.loads(cache.read_text())
    url = f"https://huggingface.co/api/datasets/{repo}/tree/{rev}/{urllib.parse.quote(path)}"
    out: list[dict] = []
    while url:
        log(f"  LIST {url}")
        body, headers = http_get(url)
        out.extend(json.loads(body))
        link = headers.get("Link") or headers.get("link") or ""
        m = re.search(r'<([^>]+)>;\s*rel="next"', link)
        url = m.group(1) if m else ""
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps(out))
    return out


def chroma_spread(im: "Image.Image") -> tuple[float, float]:
    """(chroma_spread, tint_spread) on a GRAY_GRID x GRAY_GRID sample, exactly as
    src/image_stats.rs computes them (see the module docs)."""
    rgb = im.convert("RGB")
    w, h = rgb.size

    def axis(n: int) -> list[int]:
        k = max(1, min(n, GRAY_GRID))
        return [(2 * i + 1) * n // (2 * k) for i in range(k)]

    px = rgb.load()
    lum, d1, d2 = [], [], []
    for y in axis(h):
        for x in axis(w):
            r, g, b = px[x, y]
            lum.append(0.299 * r + 0.587 * g + 0.114 * b)
            d1.append(float(r - g))
            d2.append(float(g - b))
    n = len(d1)
    raw = sum(abs(a) + abs(b) for a, b in zip(d1, d2)) / (2 * n)
    return raw, (fit_residual(lum, d1) + fit_residual(lum, d2)) / 2


def fit_residual(l: list[float], d: list[float]) -> float:
    """Mean absolute residual of d ~ a + b * l fitted by least squares (offset or gain tint)."""
    n = len(l)
    ml, md = sum(l) / n, sum(d) / n
    sll = sum((x - ml) ** 2 for x in l)
    sld = sum((x - ml) * (y - md) for x, y in zip(l, d))
    b = sld / sll if sll > 1e-9 else 0.0
    a = md - b * ml
    return sum(abs(y - (a + b * x)) for x, y in zip(l, d)) / n


def is_gray(spread: tuple[float, float]) -> bool:
    return min(spread) < GRAY_SPREAD


def box_iou(a: list[float], b: list[float]) -> float:
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def clip_box(x0: float, y0: float, x1: float, y1: float, w: int, h: int) -> list[float] | None:
    x0, x1 = max(0.0, min(float(w), x0)), max(0.0, min(float(w), x1))
    y0, y1 = max(0.0, min(float(h), y0)), max(0.0, min(float(h), y1))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        return None
    return [round(x0, 2), round(y0, 2), round(x1, 2), round(y1, 2)]


def res_tag(w: int, h: int) -> str:
    mp = w * h / 1e6
    if mp < 0.5:
        return "res:sd"
    if mp < 1.5:
        return "res:hd"
    if mp < 3.0:
        return "res:fhd"
    return "res:4mp+"


def structural_tags(cand: dict, scored: set[str] | None) -> list[str]:
    """Tags derivable from GT + size (complexity, res, small-objects, crowd, empty)."""
    objs = [o for o in cand["objects"] if not o["ignore"] and (scored is None or o["label"] in scored)]
    n = len(objs)
    areas = [(o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]) for o in objs]
    n_small = sum(1 for a in areas if a < SMALL_AREA)
    # occlusion proxy: scored boxes overlapping another scored box with IoU >= 0.3
    occluded = 0
    if 3 <= n <= 400:
        for i, a in enumerate(objs):
            if any(box_iou(a["bbox"], b["bbox"]) >= 0.3 for j, b in enumerate(objs) if j != i):
                occluded += 1
    tags = []
    if n == 0:
        tags.append("empty")
    elif n >= 10 or n_small >= 3 or occluded >= 4:
        tags.append("complexity:hard")
    elif n >= 3:
        tags.append("complexity:medium")
    else:
        tags.append("complexity:easy")
    tags.append(res_tag(cand["width"], cand["height"]))
    if n_small:
        tags.append("small-objects")
    if sum(1 for o in objs if o["label"] == "person") >= 8:
        tags.append("crowd")
    return tags


def complexity_of(tags: list[str]) -> str:
    for t in tags:
        if t.startswith("complexity:") or t == "empty":
            return t
    raise ValueError(tags)


# --------------------------------------------------------------------------------------------
# selection


def select(
    cands: list[dict],
    *,
    count: int,
    max_bytes: float,
    seed: int,
    bonus_tags: dict[str, float],
    scored: set[str] | None,
    est_size: float,
) -> list[dict]:
    """Seeded, stratified greedy pick.

    Buckets: empty (~7%), complexity easy/medium/hard (equal shares, shortfalls redistributed).
    Within a bucket the next pick maximises: static weight + label rarity + tag rarity bonus.
    Buckets are visited round-robin so a byte budget is shared fairly.
    """
    rng = random.Random(seed)
    pool = sorted(cands, key=lambda c: c["file"])
    rng.shuffle(pool)
    buckets: dict[str, list[dict]] = {k: [] for k in ("empty", "complexity:easy", "complexity:medium", "complexity:hard")}
    for c in pool:
        buckets[complexity_of(c["tags"])].append(c)

    sizes = sorted(c.get("size_hint", est_size) for c in pool)
    if sizes:  # a byte budget smaller than count * typical size lowers the effective count
        count = min(count, int(max_bytes // sizes[len(sizes) // 2]))
    n_empty = min(len(buckets["empty"]), max(1, round(count * EMPTY_FRACTION)))
    quotas = {"empty": n_empty}
    rest = count - n_empty
    names = ["complexity:easy", "complexity:medium", "complexity:hard"]
    avail = {k: len(buckets[k]) for k in names}
    remaining = list(names)
    left = rest
    for k in names:
        quotas[k] = 0
    while left > 0 and remaining:
        share = max(1, left // len(remaining))
        nxt = []
        for k in remaining:
            take = min(share, avail[k] - quotas[k], left)
            quotas[k] += take
            left -= take
            if avail[k] - quotas[k] > 0:
                nxt.append(k)
            if left <= 0:
                break
        if nxt == remaining and share == 0:
            break
        remaining = nxt

    label_imgs: Counter = Counter()
    tag_imgs: Counter = Counter()
    picked: list[dict] = []
    used_bytes = 0.0
    taken = {k: 0 for k in quotas}
    window = 400

    def gain(c: dict) -> float:
        g = c.get("weight", 0.0)
        labels = {o["label"] for o in c["objects"] if not o["ignore"] and (scored is None or o["label"] in scored)}
        g += sum(1.0 / (1 + label_imgs[lb]) for lb in labels)
        for t, w in bonus_tags.items():
            if t in c["tags"]:
                g += w / (1 + tag_imgs[t])
        return g

    order = ["complexity:hard", "complexity:medium", "complexity:easy", "empty"]
    progress = True
    while progress and len(picked) < count:
        progress = False
        for k in order:
            if taken[k] >= quotas[k] or not buckets[k]:
                continue
            best_i, best_g = -1, float("-inf")
            for i, c in enumerate(buckets[k][:window]):
                size = c.get("size_hint", est_size)
                if used_bytes + size > max_bytes:
                    continue
                g = gain(c)
                if g > best_g:
                    best_i, best_g = i, g
            if best_i < 0:
                quotas[k] = taken[k]  # budget exhausted for this bucket
                continue
            c = buckets[k].pop(best_i)
            picked.append(c)
            taken[k] += 1
            used_bytes += c.get("size_hint", est_size)
            label_imgs.update({o["label"] for o in c["objects"] if not o["ignore"]})
            tag_imgs.update(c["tags"])
            progress = True
    log(f"  selected {len(picked)} (quotas {quotas}, est {used_bytes / 1e6:.1f} MB)")
    return picked


# --------------------------------------------------------------------------------------------
# download + finalise


def fetch_image(c: dict, img_dir: Path) -> dict:
    path = img_dir / c["file"]
    data = cached(path, c["url"], c.get("sha_hint"))
    sha = hashlib.sha256(data).hexdigest()
    if c.get("sha_hint") and sha != c["sha_hint"]:
        raise RuntimeError(f"sha256 mismatch {c['file']}")
    with Image.open(io.BytesIO(data)) as im:
        im.load()
        w, h = im.size
        luma = ImageStat.Stat(im.convert("L")).mean[0]
        spread = chroma_spread(im)
    if (w, h) != (c["width"], c["height"]):
        raise RuntimeError(f"{c['file']}: decoded {w}x{h}, annotations say {c['width']}x{c['height']}")
    return {"sha256": sha, "size": len(data), "width": w, "height": h, "luma": luma, "spread": spread}


def finalise(set_id: str, meta: dict, picked: list[dict], work: Path, day_night: str) -> dict:
    img_dir = work / set_id / "images"
    images = []
    for i, c in enumerate(sorted(picked, key=lambda c: c["file"])):
        info = fetch_image(c, img_dir)
        if day_night == "night":
            dn = "night"
        else:
            dn = "night" if info["luma"] < NIGHT_LUMA or c.get("night_hint") else "day"
        tags = [dn, meta["kind_tag"]] + c["tags"]
        if meta.get("gray_tag") and is_gray(info["spread"]) and "grayscale" not in tags:
            tags.insert(1, "grayscale")
        images.append(
            {
                "file": c["file"],
                "url": c["url"],
                "sha256": info["sha256"],
                "size": info["size"],
                "width": info["width"],
                "height": info["height"],
                "tags": tags,
                "objects": c["objects"],
            }
        )
        if (i + 1) % 25 == 0:
            log(f"  fetched {i + 1}/{len(picked)}")
    total = sum(im["size"] for im in images)
    manifest = {
        "id": set_id,
        "title": meta["title"],
        "description": meta["description"].format(n=len(images), mb=total / 1e6, bytes=total),
        "license": meta["license"],
        "attribution": meta["attribution"],
        "source": meta["source"],
        "revision": meta["revision"],
    }
    if meta.get("scored_labels"):
        manifest["scored_labels"] = meta["scored_labels"]
    manifest["images"] = images
    return manifest


def write_manifest(manifest: dict) -> Path:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    path = OUT_DIR / f"{manifest['id']}.json"
    text = json.dumps(manifest, indent=1, ensure_ascii=False)
    # keep bbox arrays on one line for readable diffs
    text = re.sub(r"\[\s+([-0-9.e]+),\s+([-0-9.e]+),\s+([-0-9.e]+),\s+([-0-9.e]+)\s+\]", r"[\1, \2, \3, \4]", text)
    path.write_text(text + "\n", encoding="utf-8")
    log(f"  wrote {path} ({len(manifest['images'])} images)")
    return path


# --------------------------------------------------------------------------------------------
# BMD-45

BMD45_MAP = {
    "Hatchback": "car", "Sedan": "car", "SUV": "car", "MUV": "car", "Van": "car", "Tempo-traveller": "car",
    "Bus": "bus", "Mini-bus": "bus", "Truck": "truck",
    # LCV = light goods carriers (Tata Ace-style pickups, goods autos): COCO calls goods carriers "truck".
    "LCV": "truck",
    "Two-wheeler": "motorbike", "Bicycle": "bicycle",
}
# Ignored classes keep the nearest coco80 label so an engine matching ignore regions per class works.
BMD45_IGNORE = {"Three-wheeler": "car", "Other": "truck"}
BMD45_SCORED = ["car", "bus", "truck", "motorbike", "bicycle"]


def build_bmd45(work: Path, coco80: list[str], args) -> dict:
    d = work / "bmd45"
    ann = json.loads(cached(d / "ann.json", hf_resolve(BMD45_REPO, BMD45_REV, "BMD-45-Val/_annotations.coco.json")))
    cats = {c["id"]: c["name"] for c in ann["categories"]}
    unknown = set(cats.values()) - set(BMD45_MAP) - set(BMD45_IGNORE)
    if unknown:
        raise SystemExit(f"BMD-45: unmapped categories {unknown}")
    lfs: dict[str, dict] = {}
    for sub in ("images_000", "images_001", "images_002"):
        for e in hf_tree(BMD45_REPO, BMD45_REV, f"BMD-45-Val/{sub}", d / f"tree_{sub}.json"):
            if e.get("type") == "file" and e.get("lfs"):
                lfs[e["path"].removeprefix("BMD-45-Val/")] = e["lfs"]
    by_img: dict[int, list] = {}
    for a in ann["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    cands = []
    for im in ann["images"]:
        fn = im["file_name"]
        if fn not in lfs:
            continue
        w, h = im["width"], im["height"]
        objs = []
        for a in by_img.get(im["id"], []):
            name = cats[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            box = clip_box(x, y, x + bw, y + bh, w, h)
            if box is None:
                continue
            if name in BMD45_MAP:
                objs.append({"label": BMD45_MAP[name], "bbox": box, "ignore": bool(a.get("iscrowd", 0))})
            else:
                objs.append({"label": BMD45_IGNORE[name], "bbox": box, "ignore": True})
        objs.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["label"]))
        c = {
            "file": safe_name(fn),
            "url": hf_resolve(BMD45_REPO, BMD45_REV, f"BMD-45-Val/{fn}"),
            "width": w, "height": h, "objects": objs,
            "size_hint": lfs[fn]["size"], "sha_hint": lfs[fn]["oid"],
        }
        c["tags"] = structural_tags(c, set(BMD45_SCORED))
        cands.append(c)
    files = [c["file"] for c in cands]
    assert len(files) == len(set(files)), "BMD-45 file names are expected to be globally unique"
    log(f"  BMD-45 candidates: {len(cands)}")
    picked = select(cands, count=args.count, max_bytes=args.max_mb * 1e6, seed=args.seed,
                    bonus_tags={"small-objects": 1.0, "empty": 0.0}, scored=set(BMD45_SCORED), est_size=3.4e6)
    meta = {
        "title": "BMD-45 real CCTV (Bengaluru)",
        "kind_tag": "cctv",
        "description": (
            "{n} real CCTV frames from Bengaluru Police Safe City cameras (BMD-45 validation split, "
            "1920x1080 PNG, daytime 06:00-18:00, February 2025), {mb:.1f} MB ({bytes} bytes) in total. "
            "Vehicle-only ground truth: Hatchback/Sedan/SUV/MUV/Van/Tempo-traveller -> car, Bus/Mini-bus -> bus, "
            "Truck/LCV (light goods carrier) -> truck, Two-wheeler -> motorbike, Bicycle -> bicycle; Three-wheeler (auto-rickshaw) and Other are "
            "ignore regions. Two-wheeler and Bicycle boxes include the rider. There is no person ground truth, so only "
            "scored_labels are scored on this set. Resolution is uniform (res:fhd); the source has no per-image camera "
            "or timestamp metadata (file names are anonymised), so images are spread by seeded random sampling. "
            "Very distant vehicles are often unannotated."
        ),
        "license": "CC BY 4.0 (https://creativecommons.org/licenses/by/4.0/)",
        "attribution": (
            "BMD-45: Bengaluru Mobility Dataset, AI for Integrated Mobility (AIM) group, Indian Institute of Science "
            "(IISc), Bengaluru (Sharma, Mhatre, Gawali, Bokkasam, Kishore, Pattanaik, Rambha, Pinjari, Kovvali, "
            "Chakraborty, Rathore, Krishnapuram, Simmhan; CVPR Findings 2026). Imagery courtesy of Bengaluru Traffic "
            "Police / Bengaluru Police Safe City. Class mapping and subset selection by Blue Onyx Prism."
        ),
        "source": f"https://huggingface.co/datasets/{BMD45_REPO}",
        "revision": BMD45_REV,
        "scored_labels": BMD45_SCORED,
    }
    return finalise("bmd45-cctv", meta, picked, work, "luma")


# --------------------------------------------------------------------------------------------
# ExDark

EXDARK_NAMES = ["Bicycle", "Boat", "Bottle", "Bus", "Car", "Cat", "Chair", "Cup", "Dog", "Motorbike", "People", "Table"]
EXDARK_MAP = {
    "Bicycle": "bicycle", "Boat": "boat", "Bottle": "bottle", "Bus": "bus", "Car": "car", "Cat": "cat",
    "Chair": "chair", "Cup": "cup", "Dog": "dog", "Motorbike": "motorbike", "People": "person",
    "Table": "diningtable",
}
EXDARK_SCORED = ["person", "bicycle", "car", "motorbike", "bus", "dog", "cat"]
EXDARK_INDOOR = {"bottle", "cup", "chair", "diningtable"}


def build_exdark(work: Path, coco80: list[str], args) -> dict:
    d = work / "exdark"
    yaml_txt = cached(d / "data.yaml", hf_resolve(EXDARK_REPO, EXDARK_REV, "data/data.yaml")).decode()
    m = re.search(r"names:\s*\[(.*?)\]", yaml_txt)
    names = [s.strip().strip("'\"") for s in m.group(1).split(",")] if m else []
    if names != EXDARK_NAMES:
        raise SystemExit(f"ExDark class list changed: {names}")
    tree = hf_tree(EXDARK_REPO, EXDARK_REV, "data/test/images", d / "tree_test_images.json")
    cands = []
    scored = set(EXDARK_SCORED)
    for e in sorted(tree, key=lambda e: e["path"]):
        if e.get("type") != "file" or not e["path"].lower().endswith((".jpg", ".jpeg", ".png")):
            continue
        fn = safe_name(e["path"])
        stem = fn.rsplit(".", 1)[0]
        lbl_path = f"data/test/labels/{stem}.txt"
        try:
            txt = cached(d / "labels" / f"{stem}.txt", hf_resolve(EXDARK_REPO, EXDARK_REV, lbl_path)).decode()
        except urllib.error.HTTPError as err:
            if err.code == 404:
                continue
            raise
        w = h = 640  # Roboflow export: every image stretched to 640x640 (checked on download)
        objs = []
        for line in txt.splitlines():
            p = line.split()
            if not p:
                continue
            cls = int(p[0])
            vals = [float(v) for v in p[1:]]
            if len(vals) == 4:
                cx, cy, bw, bh = vals
                x0, y0, x1, y1 = cx - bw / 2, cy - bh / 2, cx + bw / 2, cy + bh / 2
            elif len(vals) >= 6 and len(vals) % 2 == 0:  # polygon -> enclosing box
                xs, ys = vals[0::2], vals[1::2]
                x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
            else:
                raise SystemExit(f"ExDark: bad label line in {lbl_path}: {line!r}")
            box = clip_box(x0 * w, y0 * h, x1 * w, y1 * h, w, h)
            if box is None:
                continue
            label = EXDARK_MAP[EXDARK_NAMES[cls]]
            objs.append({"label": label, "bbox": box, "ignore": label not in scored})
        objs.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["label"]))
        labels = {o["label"] for o in objs}
        weight = 0.0
        weight -= 0.5 * len(labels & EXDARK_INDOOR)  # prefer outdoor / street content
        weight += 0.5 * bool(labels & {"car", "bus", "motorbike", "bicycle"})
        weight += 1.0 * bool(labels & {"dog", "cat"})  # pets are CCTV-relevant but rare in the split
        c = {
            "file": fn,
            "url": hf_resolve(EXDARK_REPO, EXDARK_REV, e["path"]),
            "width": w, "height": h, "objects": objs, "weight": weight,
            "size_hint": e.get("lfs", {}).get("size", e.get("size", 40000)),
            "sha_hint": e.get("lfs", {}).get("oid"),
        }
        c["tags"] = structural_tags(c, scored)
        cands.append(c)
    log(f"  ExDark candidates: {len(cands)}")
    picked = select(cands, count=args.count, max_bytes=args.max_mb * 1e6, seed=args.seed,
                    bonus_tags={"small-objects": 1.0, "crowd": 1.0}, scored=scored, est_size=4e4)
    meta = {
        "title": "ExDark low-light / night",
        "kind_tag": "general",
        "description": (
            "{n} low-light photos (very low light to twilight) from the ExDark test split, {mb:.1f} MB ({bytes} bytes) "
            "in total. Images were resized (stretched) to 640x640 by the Roboflow YOLO export, so resolution is uniform "
            "(res:sd) and aspect ratios are distorted. Scored labels: person (ExDark 'People'), bicycle, car, motorbike, "
            "bus, dog, cat; boat/bottle/chair/cup/diningtable boxes are ignore regions. Selection favours outdoor "
            "street, people, vehicle and animal scenes."
        ),
        "license": (
            "BSD-3-Clause (original ExDark, https://github.com/cs-chan/Exclusively-Dark-Image-Dataset/blob/master/LICENSE); "
            "authors ask that commercial use be cleared with Dr. Chee Seng Chan. Roboflow export labelled CC BY 4.0."
        ),
        "attribution": (
            "ExDark (Exclusively Dark Image Dataset), Yuen Peng Loh and Chee Seng Chan, Universiti Malaya: 'Getting to "
            "Know Low-light Images with The Exclusively Dark Dataset', CVIU 178 (2019) 30-42, "
            "doi:10.1016/j.cviu.2018.10.010. YOLO export by Roboflow Universe workspace my-workspace-ohnbt "
            "(exclusively-dark-image v2), redistributed by dronefreak/ExDark on Hugging Face. Subset selection and "
            "class mapping by Blue Onyx Prism."
        ),
        "source": f"https://huggingface.co/datasets/{EXDARK_REPO}",
        "revision": EXDARK_REV,
        "scored_labels": EXDARK_SCORED,
    }
    return finalise("exdark-night", meta, picked, work, "night")


# --------------------------------------------------------------------------------------------
# COCO

COCO_RENAME = {
    "motorcycle": "motorbike", "airplane": "aeroplane", "couch": "sofa", "potted plant": "pottedplant",
    "dining table": "diningtable", "tv": "tvmonitor",
}
COCO_EXCLUDE = {  # indoor close-ups, food and sports gear: not CCTV-like
    "bed", "toilet", "tvmonitor", "laptop", "mouse", "remote", "keyboard", "cell phone", "microwave", "oven",
    "toaster", "sink", "refrigerator", "book", "hair drier", "toothbrush", "scissors", "teddy bear", "sandwich",
    "hot dog", "pizza", "donut", "cake", "banana", "apple", "orange", "broccoli", "carrot", "bowl", "fork", "knife",
    "spoon", "wine glass", "cup", "skis", "snowboard", "surfboard", "tennis racket", "baseball bat",
    "baseball glove", "sports ball", "aeroplane", "giraffe", "zebra", "elephant", "bear", "train",
}
COCO_STREET = {"car", "bus", "truck", "motorbike", "bicycle", "traffic light", "stop sign", "fire hydrant",
               "parking meter", "bench"}
COCO_REVISION = f"val2017; annotations_trainval2017.zip sha256 {COCO_ANN_ZIP_SHA256}; tags r2 (grayscale)"
COCO_GRAY_NOTE = (
    " Monochrome images (chroma spread below " + str(GRAY_SPREAD) + "/255, the service's IR test) are tagged "
    "grayscale."
)
NIGHT_WORDS = re.compile(r"\b(night|nighttime|night-time|dark|darkness|evening|dusk|lit up|neon)\b", re.I)


def build_coco(work: Path, coco80: list[str], args) -> dict:
    d = work / "coco"
    zpath = d / "annotations_trainval2017.zip"
    cached(zpath, COCO_ANN_ZIP, COCO_ANN_ZIP_SHA256)
    with zipfile.ZipFile(zpath) as z:
        inst = json.loads(z.read("annotations/instances_val2017.json"))
        caps = json.loads(z.read("annotations/captions_val2017.json"))
    cats = {}
    for c in inst["categories"]:
        name = COCO_RENAME.get(c["name"], c["name"])
        if name not in coco80:
            raise SystemExit(f"COCO category {c['name']!r} -> {name!r} not in coco80")
        cats[c["id"]] = name
    night_hint = set()
    for c in caps["annotations"]:
        if NIGHT_WORDS.search(c["caption"]):
            night_hint.add(c["image_id"])
    by_img: dict[int, list] = {}
    for a in inst["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    focus = set(CCTV_FOCUS)
    cands = []
    for im in inst["images"]:
        anns = by_img.get(im["id"], [])
        w, h = im["width"], im["height"]
        objs = []
        for a in anns:
            x, y, bw, bh = a["bbox"]
            box = clip_box(x, y, x + bw, y + bh, w, h)
            if box is None:
                continue
            crowd = bool(a["iscrowd"])
            obj = {"label": cats[a["category_id"]], "bbox": box, "ignore": crowd}
            if crowd:
                obj["crowd"] = True  # matched by intersection over the detection (pycocotools' crowd rule)
            objs.append(obj)
        objs.sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["label"]))
        labels = {o["label"] for o in objs}
        if objs:
            if labels & COCO_EXCLUDE or not {o["label"] for o in objs if not o["ignore"]} & focus:
                continue
        weight = 1.0 * bool(labels & COCO_STREET) + 0.5 * bool(labels & {"dog", "cat", "horse", "bird"})
        fn = safe_name(im["file_name"])
        c = {
            "file": fn, "url": f"{COCO_BASE}/val2017/{fn}", "width": w, "height": h, "objects": objs,
            "weight": weight, "night_hint": im["id"] in night_hint,
        }
        c["tags"] = structural_tags(c, None)
        if c["night_hint"]:
            c["tags"].append("night-hint")
        cands.append(c)
    log(f"  COCO candidates: {len(cands)} ({sum(c['night_hint'] for c in cands)} with night captions)")
    picked = select(cands, count=args.count, max_bytes=args.max_mb * 1e6, seed=args.seed,
                    bonus_tags={"night-hint": 3.0, "small-objects": 1.0, "crowd": 1.5}, scored=None, est_size=1.7e5)
    for c in picked:
        c["tags"] = [t for t in c["tags"] if t != "night-hint"]
    meta = {
        "title": "COCO val2017 CCTV-relevant subset",
        "kind_tag": "general",
        "description": (
            "{n} COCO val2017 images with CCTV-relevant content (street scenes, people from single to crowds, "
            "vehicles, bicycles, motorbikes, animals, yards and parking), {mb:.1f} MB ({bytes} bytes) in total. "
            "All 80 COCO labels are scored (coco80 spellings); iscrowd regions are ignore. CCTV focus classes: "
            + ", ".join(CCTV_FOCUS) + ". COCO images are at most 640 px on the long side (res:sd). Day/night is "
            "estimated from mean luminance (< " + str(int(NIGHT_LUMA)) + "/255) or a night/dark caption. "
            "Empty images are val2017 images with no annotations." + COCO_GRAY_NOTE
        ),
        "license": (
            "Annotations: CC BY 4.0 (COCO Consortium). Images: individual Flickr licences as listed in "
            "instances_val2017.json (CC BY / BY-SA / BY-NC / BY-NC-SA / BY-ND / BY-NC-ND 2.0 and others); "
            "images are fetched from the COCO host, not redistributed."
        ),
        "attribution": (
            "Microsoft COCO: Common Objects in Context (Lin et al., ECCV 2014), https://cocodataset.org; images by "
            "their Flickr authors. Subset selection and label mapping by Blue Onyx Prism."
        ),
        "source": "https://cocodataset.org (val2017, annotations_trainval2017.zip)",
        "revision": COCO_REVISION,
        "gray_tag": True,
    }
    return finalise("coco-cctv", meta, picked, work, "luma")


def retag_grayscale(set_id: str, work: Path) -> dict:
    """Recompute the `grayscale` tag of an existing manifest from its pinned images (downloaded
    into <work>/<set>/images and checked against the manifest's SHA-256); selection, ground truth
    and other tags are unchanged. Bumps coco-cctv's revision (its tags changed)."""
    path = OUT_DIR / f"{set_id}.json"
    m = json.loads(path.read_text(encoding="utf-8"))
    img_dir = work / set_id / "images"
    n_gray = 0
    for i, im in enumerate(m["images"]):
        data = cached(img_dir / safe_name(im["file"]), im["url"], im["sha256"])
        with Image.open(io.BytesIO(data)) as pil:
            pil.load()
            spread = chroma_spread(pil)
        tags = [t for t in im["tags"] if t != "grayscale"]
        if is_gray(spread):
            tags.insert(1, "grayscale")
            n_gray += 1
            log(f"  grayscale {im['file']} spread {spread[0]:.2f} / tint-corrected {spread[1]:.2f}")
        im["tags"] = tags
        if (i + 1) % 25 == 0:
            log(f"  checked {i + 1}/{len(m['images'])}")
    if set_id == "coco-cctv":
        m["revision"] = COCO_REVISION
        if COCO_GRAY_NOTE.strip() not in m["description"]:
            m["description"] += COCO_GRAY_NOTE
    log(f"  {n_gray} of {len(m['images'])} images are grayscale")
    return m


# --------------------------------------------------------------------------------------------
# exdark-ir (no images of its own)


def build_exdark_ir(work: Path, coco80: list[str], args) -> dict:
    src = json.loads((OUT_DIR / "exdark-night.json").read_text(encoding="utf-8"))
    return {
        "id": "exdark-ir",
        "title": "ExDark grayscale (NIR proxy)",
        "description": (
            f"The {len(src['images'])} exdark-night images and ground truth, converted after decode to BT.601 luma "
            "replicated to three channels (\"transform\": \"grayscale\"). A rough proxy for night IR (near-infrared) "
            "monochrome, NOT real IR: IR illumination changes contrast and reflectance (foliage and skin turn bright, "
            "fabric differs), which grayscale visible-light photos do not show. Same scored labels and ignore regions as "
            "exdark-night; the images are those of exdark-night (downloaded once, into its folder)."
        ),
        "license": src["license"],
        "attribution": src["attribution"],
        "source": src["source"],
        "revision": f"exdark-night {src['revision']}; transform grayscale (BT.601) v1",
        "scored_labels": src.get("scored_labels"),
        "transform": "grayscale",
        "images_from": "exdark-night",
        "images": [],
    }


# --------------------------------------------------------------------------------------------
# nir-night (LILA BC camera traps)

LILA = "https://storage.googleapis.com/public-datasets-lila"
CCT_BBOX_URL = f"{LILA}/caltechcameratraps/labels/caltech_bboxes_20200316.json"
CCT_BBOX_SHA256 = "9cc208db67b7dcf3fa403920c203a416527628f76a6b838371ab654478bf2ea3"
CCT_LABELS_URL = f"{LILA}/caltechcameratraps/labels/caltech_camera_traps.json.zip"
CCT_LABELS_SHA256 = "976bd9bc9380df7a03018ac70b5340bd2ff81d79a99271058f026a539450d1d6"
CCT_IMAGES = f"{LILA}/caltech-unzipped/cct_images/"
ENA24_URL = f"{LILA}/ena24/ena24.json"
ENA24_SHA256 = "8b426c87c948ab91dda10e5b210fc469db8b0c095a15cc7c03bb4ddceffa3160"
ENA24_IMAGES = f"{LILA}/ena24/images/"
NACTI_BBOX_URL = f"{LILA}/nacti/nacti_20230920_bboxes.zip"
NACTI_BBOX_SHA256 = "e95ef551f8a4b2bad2ad73f7ecccc40acabf4139ecda61522aa5d583fbfbec34"
NACTI_IMAGES = f"{LILA}/nacti-unzipped/"

NIR_SCORED = ["person", "car", "dog", "cat", "raccoon"]
# Non-COCO animals: must match WILDLIFE in src/benchmark/metrics.rs (a COCO model's dog/cat box on
# one is neither a hit nor a false positive).
WILDLIFE = {
    "animal", "armadillo", "badger", "bear", "beaver", "boar", "bobcat", "chipmunk", "cottontail", "cougar",
    "coyote", "deer", "elk", "ermine", "fox", "groundhog", "hare", "jackrabbit", "javelina", "lion", "lynx",
    "marmot", "marten", "mink", "moose", "opossum", "otter", "peccary", "porcupine", "possum", "puma", "rabbit",
    "raccoon", "rodent", "skunk", "squirrel", "weasel", "wolf", "wolverine", "woodchuck",
}
CCT_MAP = {"car": "car", "dog": "dog", "cat": "cat", "raccoon": "raccoon"}
CCT_IGNORE = {
    "opossum": "opossum", "coyote": "coyote", "deer": "deer", "rabbit": "rabbit", "bobcat": "bobcat",
    "squirrel": "squirrel", "rodent": "rodent", "skunk": "skunk", "fox": "fox", "mountain_lion": "mountain lion",
    "badger": "badger",
}
CCT_SKIP = {"bird", "lizard", "insect", "bat", "cow", "pig"}  # COCO-like or tiny: left out of the set
ENA24_MAP = {"Dog": "dog", "Domestic Cat": "cat", "Northern Raccoon": "raccoon", "Vehicle": "car"}
ENA24_IGNORE = {
    "American Black Bear": "black bear", "Virginia Opossum": "opossum", "Grey Fox": "fox", "Red Fox": "fox",
    "White_Tailed_Deer": "deer", "Coyote": "coyote", "Eastern Fox Squirrel": "squirrel",
    "Eastern Gray Squirrel": "squirrel", "Eastern Cottontail": "rabbit", "Bobcat": "bobcat",
    "Eastern Chipmunk": "chipmunk", "Striped Skunk": "skunk", "Woodchuck": "woodchuck",
}
ENA24_SKIP = {"Human", "American Crow", "Chicken", "Wild Turkey", "Bird", "Horse"}
NIR_QUOTA = {"car": 30, "dog": 30, "cat": 30, "raccoon": 30}
NIR_WILDLIFE_ONLY = 20
NIR_NEGATIVES = 80
# ENA24 (and NACTI frames without EXIF time) have no capture time, and their cameras also record
# monochrome in overcast daylight: such frames must also be darker than this mean luma (0..255).
# IR-flash night frames measured 24..117 (median ~70); the daytime monochrome ones 120..170.
NO_TIME_MAX_LUMA = 110.0
NIR_GROUP_ORDER = ["car", "dog", "cat", "raccoon", "wildlife-only", "empty"]


def is_wildlife(label: str) -> bool:
    l = label.lower().replace("_", " ")
    return l != "teddy bear" and any(w in WILDLIFE for w in re.split(r"[ -]", l) if w)


def night_hour(date: str | None) -> bool | None:
    """20:00..04:59 from a 'YYYY-MM-DD HH:MM:SS' capture time; None when absent / unparsable."""
    m = re.match(r"^\d{4}-\d{2}-\d{2}[ T](\d{2}):", date or "")
    if not m:
        return None
    h = int(m.group(1))
    return h >= 20 or h < 5


def cct_candidates(d: Path) -> tuple[list[dict], list[dict]]:
    """(positive / wildlife candidates from the box file, empty candidates from the image labels)."""
    bb = json.loads(cached(d / "caltech_bboxes_20200316.json", CCT_BBOX_URL, CCT_BBOX_SHA256))
    cats = {c["id"]: c["name"] for c in bb["categories"]}
    unknown = set(cats.values()) - set(CCT_MAP) - set(CCT_IGNORE) - CCT_SKIP - {"empty"}
    if unknown:
        raise SystemExit(f"CCT: unmapped categories {unknown}")
    by_img: dict[str, list] = {}
    for a in bb["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    pos = []
    for im in bb["images"]:
        anns = by_img.get(im["id"], [])
        names = {cats[a["category_id"]] for a in anns}
        if not anns or names & CCT_SKIP or names == {"empty"} or "empty" in names:
            continue
        if night_hour(im.get("date_captured")) is not True:
            continue
        w, h = im["width"], im["height"]
        objs = []
        for a in anns:
            name = cats[a["category_id"]]
            if "bbox" not in a:
                objs = None
                break
            x, y, bw, bh = a["bbox"]
            box = clip_box(x, y, x + bw, y + bh, w, h)
            if box is None:
                continue
            if name in CCT_MAP:
                objs.append({"label": CCT_MAP[name], "bbox": box, "ignore": False})
            else:
                objs.append({"label": CCT_IGNORE[name], "bbox": box, "ignore": True})
        if not objs:
            continue
        pos.append({
            "file": "cct-" + safe_name(im["file_name"]), "url": CCT_IMAGES + safe_name(im["file_name"]),
            "width": w, "height": h, "objects": objs, "source": "cct", "location": f"cct:{im['location']}",
            "seq": im.get("seq_id"), "date": im.get("date_captured"),
        })
    z = zipfile.ZipFile(io.BytesIO(cached(d / "caltech_camera_traps.json.zip", CCT_LABELS_URL, CCT_LABELS_SHA256)))
    lab = json.loads(z.read("caltech_images_20210113.json"))
    lcats = {c["id"]: c["name"] for c in lab["categories"]}
    labels_of: dict[str, set] = {}
    for a in lab["annotations"]:
        labels_of.setdefault(a["image_id"], set()).add(lcats[a["category_id"]])
    neg = []
    for im in lab["images"]:
        if labels_of.get(im["id"]) != {"empty"} or night_hour(im.get("date_captured")) is not True:
            continue
        neg.append({
            "file": "cct-" + safe_name(im["file_name"]), "url": CCT_IMAGES + safe_name(im["file_name"]),
            "width": im["width"], "height": im["height"], "objects": [], "source": "cct",
            "location": f"cct:{im['location']}", "seq": im.get("seq_id"), "date": im.get("date_captured"),
        })
    return pos, neg


def ena24_candidates(d: Path) -> list[dict]:
    """ENA24 has no capture times: night is decided by the grayscale (IR) test alone. Images
    with a Human box (removed from the public bucket) and bird/horse images are left out."""
    e = json.loads(cached(d / "ena24.json", ENA24_URL, ENA24_SHA256))
    cats = {c["id"]: c["name"] for c in e["categories"]}
    unknown = set(cats.values()) - set(ENA24_MAP) - set(ENA24_IGNORE) - ENA24_SKIP
    if unknown:
        raise SystemExit(f"ENA24: unmapped categories {unknown}")
    by_img: dict[str, list] = {}
    for a in e["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    out = []
    for im in e["images"]:
        anns = by_img.get(im["id"], [])
        if not anns or {cats[a["category_id"]] for a in anns} & ENA24_SKIP:
            continue
        w, h = im["width"], im["height"]
        objs = []
        for a in anns:
            name = cats[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            box = clip_box(x, y, x + bw, y + bh, w, h)
            if box is None:
                continue
            if name in ENA24_MAP:
                objs.append({"label": ENA24_MAP[name], "bbox": box, "ignore": False})
            else:
                objs.append({"label": ENA24_IGNORE[name], "bbox": box, "ignore": True})
        if not objs:
            continue
        n = int(im["id"]) if str(im["id"]).isdigit() else 0
        out.append({
            "file": "ena24-" + safe_name(im["file_name"]), "url": ENA24_IMAGES + safe_name(im["file_name"]),
            "width": w, "height": h, "objects": objs, "source": "ena24",
            # No camera ids: consecutive numbers are bursts of one camera. Two overlapping buckets
            # (n // 10 and (n + 5) // 10) make any two ids less than 5 apart collide.
            "location": "ena24", "seq": f"ena24:{n // 10}", "seq2": f"ena24b:{(n + 5) // 10}", "date": None,
        })
    return out


def nacti_candidates(d: Path, persons: bool) -> list[dict]:
    """NACTI boxes (vehicle / animal / person); no capture times. Person frames are left out
    unless `persons` (see --nacti-persons)."""
    z = zipfile.ZipFile(io.BytesIO(cached(d / "nacti_20230920_bboxes.zip", NACTI_BBOX_URL, NACTI_BBOX_SHA256)))
    n = json.loads(z.read("nacti_20230920_bboxes.json"))
    cats = {c["id"]: c["name"] for c in n["categories"]}
    if set(cats.values()) - {"vehicle", "animal", "person", "group", "empty"}:
        raise SystemExit(f"NACTI: unexpected categories {set(cats.values())}")
    by_img: dict[str, list] = {}
    for a in n["annotations"]:
        by_img.setdefault(a["image_id"], []).append(a)
    out = []
    for im in n["images"]:
        anns = by_img.get(im["id"], [])
        names = {cats[a["category_id"]] for a in anns}
        if not anns or names & {"group", "empty"} or ("person" in names and not persons):
            continue
        fn = im["file_name"]
        if not re.fullmatch(r"part[0-9]/sub[0-9]{3}/[A-Za-z0-9][A-Za-z0-9._-]{0,200}", fn):
            continue
        w, h = im["width"], im["height"]
        objs = []
        for a in anns:
            name = cats[a["category_id"]]
            x, y, bw, bh = a["bbox"]
            box = clip_box(x, y, x + bw, y + bh, w, h)
            if box is None:
                continue
            if name == "vehicle":
                objs.append({"label": "car", "bbox": box, "ignore": False})
            elif name == "person":
                objs.append({"label": "person", "bbox": box, "ignore": False})
            else:
                objs.append({"label": "animal", "bbox": box, "ignore": True})
        if not objs:
            continue
        base = safe_name(fn)
        out.append({
            "file": "nacti-" + base.rsplit(".", 1)[0] + ".jpg" if base.lower().endswith(".jpg") else "nacti-" + base,
            "url": NACTI_IMAGES + urllib.parse.quote(fn), "width": w, "height": h, "objects": objs,
            "source": "nacti", "location": f"nacti:{im.get('location', '?')}",
            "seq": "nacti:" + base.rsplit("_", 1)[0], "date": None,
        })
    return out


def nir_group(c: dict) -> str:
    scored = {o["label"] for o in c["objects"] if not o["ignore"]}
    if not c["objects"]:
        return "empty"
    for g in ("car", "dog", "cat", "raccoon", "person"):
        if g in scored:
            return g
    return "wildlife-only"


def round_robin(cands: list[dict], seed: int) -> list[dict]:
    """Seeded order that visits sources in turn and, within a source, its locations in turn."""
    rng = random.Random(seed)
    by_src: dict[str, dict[str, list]] = {}
    for c in sorted(cands, key=lambda c: c["file"]):
        by_src.setdefault(c["source"], {}).setdefault(c["location"], []).append(c)
    src_iters = []
    for src in sorted(by_src):
        locs = by_src[src]
        for lst in locs.values():
            rng.shuffle(lst)
        keys = sorted(locs)
        rng.shuffle(keys)
        order = []
        while any(locs[k] for k in keys):
            for k in keys:
                if locs[k]:
                    order.append(locs[k].pop())
        src_iters.append(order)
    out = []
    while any(src_iters):
        for it in src_iters:
            if it:
                out.append(it.pop(0))
    return out


class NirFetcher:
    """Downloads candidates into <work>/nir-night/images and records their pixel checks in
    checks.json, keyed by GRAY_RULE (reruns reuse it; a rule change re-checks the cached files)."""

    def __init__(self, work: Path):
        self.dir = work / "nir-night" / "images"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.db_path = work / "nir-night" / "checks.json"
        self.db = json.loads(self.db_path.read_text()) if self.db_path.exists() else {}
        self.fetched = 0
        self.exif_path = work / "nir-night" / "exif.json"
        self.exif = json.loads(self.exif_path.read_text()) if self.exif_path.exists() else {}

    def prefetch(self, cands: list[dict]) -> None:
        """Download candidates that have no current check yet, 16 at a time (errors are left
        for `check` to record)."""
        from concurrent.futures import ThreadPoolExecutor

        todo = [c for c in cands if not (self.dir / c["file"]).exists()
                and not (self.db.get(c["file"], {}).get("error", "").startswith("HTTP"))]

        def get(c: dict) -> None:
            try:
                if self.exif_day(c):
                    return
                cached(self.dir / c["file"], c["url"])
            except Exception:  # noqa: BLE001 - recorded by check()
                pass

        with ThreadPoolExecutor(16) as ex:
            list(ex.map(get, todo))

    def exif_day(self, c: dict) -> bool:
        """For candidates without a capture time in the metadata (NACTI), the EXIF capture time
        from the first 64 KB of the file: True when it is daytime (05:00..19:59), so the image
        is rejected without a full download. Cached in the check record's `exif_time`."""
        if c.get("date") or c["source"] != "nacti":
            return False
        key = c["file"]
        if key not in self.exif:
            t = None
            try:
                req = urllib.request.Request(c["url"], headers={"User-Agent": USER_AGENT, "Range": "bytes=0-65535"})
                with urllib.request.urlopen(req, timeout=60) as resp:
                    head = resp.read(65536)
                with Image.open(io.BytesIO(head)) as im:
                    e = im.getexif()
                    t = e.get_ifd(0x8769).get(36867) or e.get(306)
            except Exception:  # noqa: BLE001 - untrusted data; no time = decided by pixels
                t = None
            self.exif[key] = t if isinstance(t, str) else None
        t = self.exif[key]
        c["date"] = t.replace(":", "-", 2) if t else None
        return night_hour(c["date"]) is False

    def save(self) -> None:
        self.db_path.write_text(json.dumps(self.db, indent=0, sort_keys=True))
        self.exif_path.write_text(json.dumps(self.exif, indent=0, sort_keys=True))

    def check(self, c: dict) -> dict | None:
        """The image's {sha256, size, width, height, spread, luma, ok}, or None (404 etc.)."""
        if self.exif_day(c):
            return {"ok": False, "error": f"daytime by EXIF ({c['date']})"}
        rec = self.db.get(c["file"])
        path = self.dir / c["file"]
        if rec is not None and rec.get("rule") == GRAY_RULE and (rec.get("error") or path.exists()):
            return rec
        if rec is not None and rec.get("error", "").startswith("HTTP"):
            return rec
        try:
            data = cached(path, c["url"])
        except (urllib.error.HTTPError, RuntimeError) as e:
            log(f"  skip {c['file']}: {e}")
            self.db[c["file"]] = {"ok": False, "error": str(e), "rule": GRAY_RULE}
            return self.db[c["file"]]
        self.fetched += 1
        try:
            with Image.open(io.BytesIO(data)) as im:
                im.load()
                w, h = im.size
                spread = chroma_spread(im)
                luma = ImageStat.Stat(im.convert("L")).mean[0]
        except Exception as e:  # noqa: BLE001 - untrusted data
            path.unlink(missing_ok=True)
            self.db[c["file"]] = {"ok": False, "error": f"decode: {e}", "rule": GRAY_RULE}
            return self.db[c["file"]]
        ok = (w, h) == (c["width"], c["height"]) and is_gray(spread)
        rec = {
            "sha256": hashlib.sha256(data).hexdigest(), "size": len(data), "width": w, "height": h,
            "spread": [round(spread[0], 3), round(spread[1], 3)], "luma": round(luma, 2), "ok": ok,
            "rule": GRAY_RULE,
        }
        if (w, h) != (c["width"], c["height"]):
            rec["error"] = f"decoded {w}x{h}, metadata says {c['width']}x{c['height']}"
        # Rejected images stay in the cache folder, so a rule change re-checks them offline.
        self.db[c["file"]] = rec
        if self.fetched % 20 == 0:
            self.save()
        return rec


def pick_nir(pool: list[dict], fetch: NirFetcher, *, want, max_tries: int, used: set, seqs: set,
             exclude: set, label: str) -> list[dict]:
    """Walk `pool` (already in seeded round-robin order) and accept candidates that pass the
    pixel checks until `want(accepted)` is False; one image per sequence."""
    out: list[dict] = []
    tries = 0
    prefetched = 0
    for i, c in enumerate(pool):
        if not want(out) or tries >= max_tries:
            break
        keys = [k for k in (c.get("seq"), c.get("seq2")) if k]
        if c["file"] in used or c["file"] in exclude or any(k in seqs for k in keys):
            continue
        if i >= prefetched:
            # Download the next candidates in parallel; decisions stay sequential (deterministic).
            batch = [x for x in pool[i:i + 96] if x["file"] not in used and x["file"] not in exclude]
            fetch.prefetch(batch)
            prefetched = i + 96
        tries += 1
        rec = fetch.check(c)
        if not rec or not rec.get("ok"):
            continue
        if not c.get("date") and rec["luma"] > NO_TIME_MAX_LUMA:
            continue  # no capture time: a bright monochrome frame is likely overcast daylight
        c = dict(c, **{k: rec[k] for k in ("sha256", "size", "width", "height", "spread", "luma")})
        out.append(c)
        used.add(c["file"])
        seqs.update(keys)
    log(f"  {label}: accepted {len(out)} after {tries} checks")
    return out


def write_review(review: Path, picked: list[dict], work: Path) -> Path:
    """Contact sheet (thumbnails with the ground-truth boxes) of every picked image, candidate
    negatives first, for manual screening."""
    from PIL import ImageDraw

    review.mkdir(parents=True, exist_ok=True)
    thumbs = review / "thumbs"
    thumbs.mkdir(exist_ok=True)
    rows = {g: [] for g in ["empty", "wildlife-only", "car", "dog", "cat", "raccoon", "person"]}
    for c in picked:
        src = work / "nir-night" / "images" / c["file"]
        with Image.open(src) as im:
            im = im.convert("RGB")
            scale = 480 / im.width
            th = im.resize((480, max(1, round(im.height * scale))))
        dr = ImageDraw.Draw(th)
        for o in c["objects"]:
            b = [v * scale for v in o["bbox"]]
            dr.rectangle(b, outline=(255, 200, 0) if o["ignore"] else (0, 255, 0), width=2)
            dr.text((b[0] + 3, b[1] + 2), o["label"], fill=(255, 255, 0))
        name = c["file"].rsplit(".", 1)[0] + ".jpg"
        th.save(thumbs / name, quality=80)
        rows[c["group"]].append((c, name))
    import html as _html

    parts = [
        "<!doctype html><meta charset=utf-8><title>nir-night review</title>",
        "<style>body{font:13px sans-serif;background:#222;color:#ddd}div.g{display:flex;flex-wrap:wrap}"
        "figure{margin:4px;width:480px}img{width:480px}figcaption{word-break:break-all}</style>",
        "<h1>nir-night review</h1><p>Check the <b>empty</b> (hard negative) frames for unlabelled people or "
        "animals, and the others for missing boxes (green = scored, orange = ignore). Put the file names to drop "
        "in an exclude file (one per line) and re-run build_bench_sets.py --set nir-night --exclude FILE.</p>",
    ]
    for g, items in rows.items():
        if not items:
            continue
        parts.append(f"<h2>{_html.escape(g)} ({len(items)})</h2><div class=g>")
        for c, name in items:
            cap = f"{c['file']} | {c['location']} | {c.get('date') or 'no time'} | spread {c['spread'][0]:.2f}/{c['spread'][1]:.2f} | luma {c['luma']:.0f}"
            parts.append(f"<figure><img loading=lazy src='thumbs/{_html.escape(name)}'><figcaption>{_html.escape(cap)}</figcaption></figure>")
        parts.append("</div>")
    out = review / "index.html"
    out.write_text("\n".join(parts), encoding="utf-8")
    (review / "selected.txt").write_text("\n".join(f"{c['group']}\t{c['file']}" for c in picked) + "\n")
    log(f"  review sheet: {out}")
    return out


def build_nir_night(work: Path, coco80: list[str], args) -> dict:
    d = work / "lila"
    exclude: set[str] = set()
    if args.exclude:
        exclude = {l.split("\t")[-1].strip() for l in args.exclude.read_text().splitlines() if l.strip() and not l.startswith("#")}
        log(f"  excluding {len(exclude)} file(s) from {args.exclude}")
    cct_pos, cct_neg = cct_candidates(d)
    ena = ena24_candidates(d)
    nacti = nacti_candidates(d, args.nacti_persons)
    pos = cct_pos + ena + nacti
    log(f"  candidates: CCT night boxed {len(cct_pos)}, CCT night empty {len(cct_neg)}, ENA24 {len(ena)}, NACTI {len(nacti)}")
    fetch = NirFetcher(work)
    used: set[str] = set()
    seqs: set[str] = set()
    picked: list[dict] = []
    budget = args.max_mb * 1e6
    try:
        classes = list(NIR_QUOTA) + (["person"] if args.nacti_persons else [])
        quota = dict(NIR_QUOTA, **({"person": 30} if args.nacti_persons else {}))
        # Rarest class first, so images with several classes count toward all of them.
        for cls in sorted(classes, key=lambda k: sum(1 for c in pos if any(o["label"] == k and not o["ignore"] for o in c["objects"]))):
            have = sum(1 for c in picked if any(o["label"] == cls and not o["ignore"] for o in c["objects"]))
            pool = round_robin([c for c in pos if any(o["label"] == cls and not o["ignore"] for o in c["objects"])],
                               args.seed + len(cls))
            picked += pick_nir(pool, fetch, want=lambda out, n=quota[cls] - have: len(out) < n,
                               max_tries=quota[cls] * (60 if cls == "car" else 12), used=used, seqs=seqs,
                               exclude=exclude, label=cls)
        wild = [c for c in pos if c["objects"] and all(o["ignore"] for o in c["objects"])]
        picked += pick_nir(round_robin(wild, args.seed + 101), fetch, want=lambda out: len(out) < NIR_WILDLIFE_ONLY,
                           max_tries=NIR_WILDLIFE_ONLY * 8, used=used, seqs=seqs, exclude=exclude, label="wildlife-only")
        picked += pick_nir(round_robin(cct_neg, args.seed + 202), fetch, want=lambda out: len(out) < NIR_NEGATIVES,
                           max_tries=NIR_NEGATIVES * 8, used=used, seqs=seqs, exclude=exclude, label="empty")
    finally:
        fetch.save()
    total = sum(c["size"] for c in picked)
    if total > budget:
        raise SystemExit(f"nir-night: {total / 1e6:.1f} MB exceeds the {args.max_mb} MB budget")
    for c in picked:
        c["group"] = nir_group(c)
        tags = ["night", "ir", "grayscale", "cctv", "camera-trap"]
        st = structural_tags(c, set(NIR_SCORED))
        if c["group"] == "empty":
            tags += st + ["hard-negative"]
        elif c["group"] == "wildlife-only":
            # No scored object, but not an empty frame: animals the scored labels leave out.
            tags += [t for t in st if t != "empty"] + ["wildlife-only"]
        else:
            tags += st
        c["tags"] = list(dict.fromkeys(tags))
        c["objects"].sort(key=lambda o: (o["bbox"][1], o["bbox"][0], o["label"]))
    picked.sort(key=lambda c: (NIR_GROUP_ORDER.index(c["group"]) if c["group"] in NIR_GROUP_ORDER else 99, c["file"]))
    # Interleave the groups so `--max-images N` (evenly spread) keeps every group in proportion.
    picked = interleave(picked)
    if args.review_dir:
        write_review(args.review_dir, picked, work)
    images = [{
        "file": c["file"], "url": c["url"], "sha256": c["sha256"], "size": c["size"], "width": c["width"],
        "height": c["height"], "tags": c["tags"], "objects": c["objects"],
    } for c in picked]
    groups = Counter(c["group"] for c in picked)
    srcs = Counter(c["source"] for c in picked)
    persons = args.nacti_persons
    return {
        "id": "nir-night",
        "title": "Night IR camera traps (LILA BC)",
        "description": (
            f"{len(images)} real night infrared (monochrome) frames from fixed outdoor camera traps, {total / 1e6:.1f} MB "
            f"({total} bytes) in total: {groups.get('car', 0)} with vehicles, {groups.get('dog', 0)} dogs, "
            f"{groups.get('cat', 0)} cats, {groups.get('raccoon', 0)} raccoons (by first scored class), "
            f"{groups.get('wildlife-only', 0)} with only other wildlife (ignore regions) and {groups.get('empty', 0)} "
            "empty frames (hard negatives: vegetation, rocks, IR glare). Sources: Caltech Camera Traps "
            f"({srcs.get('cct', 0)}; capture time 20:00-05:00), ENA24-detection ({srcs.get('ena24', 0)}) and NACTI "
            f"({srcs.get('nacti', 0)}); every frame passes the grayscale (IR) test (chroma spread < {GRAY_SPREAD}/255). "
            + ("NACTI person boxes are included (person is scored). " if persons else
               "LILA removed images of people from CCT and ENA24 and this build leaves out NACTI's person frames, so "
               "there is no person ground truth: person is scored only as false positives (every person box is a "
               "false positive). ")
            + "Vehicle (ENA24, NACTI) and car (CCT) boxes are scored as car. Raccoon is scored only for models that "
            "have the class; for other models raccoons, like the other wildlife boxes (opossum, skunk, coyote, deer, "
            "...), are ignore regions for their animal classes (a cat box on a raccoon is neither right nor wrong). "
            "Camera-trap viewpoints are low and close (not a mounted CCTV view). LILA's empty labels can miss small "
            "animals; the empty frames were screened by hand."
        ),
        "license": "CDLA-Permissive-1.0 (Community Data License Agreement - Permissive, https://cdla.io/permissive-1-0/)",
        "attribution": (
            "Images and annotations from LILA BC (Labeled Information Library of Alexandria: Biology and Conservation, "
            "https://lila.science). Caltech Camera Traps: Sara Beery, Grant Van Horn, Pietro Perona, 'Recognition in "
            "Terra Incognita', ECCV 2018 (images courtesy of the USGS and NPS; Justin Brown, Erin Boydston). ENA24-detection: "
            "Hayder Yousif, Roland Kays, Zhihai He, 'Dynamic Programming Selection of Object Proposals for Sequence-Level "
            "Animal Species Classification in the Wild', IEEE TCSVT 2019. NACTI: Tabak et al., 'Machine learning to "
            "classify animal species in camera trap images: Applications in ecology', Methods in Ecology and Evolution "
            "10(4):585-590, 2019. Subset selection, night/IR filtering and label mapping by Blue Onyx Prism."
        ),
        "source": "https://lila.science (caltech-camera-traps, ena24detection, nacti)",
        "revision": (
            f"cct bboxes {CCT_BBOX_SHA256[:12]}, cct labels {CCT_LABELS_SHA256[:12]}, ena24 {ENA24_SHA256[:12]}, "
            f"nacti bboxes {NACTI_BBOX_SHA256[:12]}; seed {args.seed}; excluded {len(exclude)}"
            + ("; persons" if persons else "")
        ),
        "scored_labels": NIR_SCORED,
        "images": images,
    }


def interleave(items: list[dict]) -> list[dict]:
    """Spread the groups evenly through the list (largest remainder), keeping order inside a group."""
    groups: dict[str, list] = {}
    for c in items:
        groups.setdefault(c["group"], []).append(c)
    n = len(items)
    keyed = []
    for g, lst in groups.items():
        k = len(lst)
        for i, c in enumerate(lst):
            keyed.append(((i + 0.5) / k, NIR_GROUP_ORDER.index(g) if g in NIR_GROUP_ORDER else 99, c))
    keyed.sort(key=lambda t: (t[0], t[1]))
    assert len(keyed) == n
    return [c for _, _, c in keyed]


# --------------------------------------------------------------------------------------------
# validation / stats


def validate(path: Path, coco80: list[str], work: Path | None, head: bool) -> dict:
    # Explicit checks that append to `errors` (not `assert`: `python -O` would skip them).
    m = json.loads(path.read_text(encoding="utf-8"))
    errors: list[str] = []
    for k in ("id", "title", "description", "license", "attribution", "source", "revision", "images"):
        if k not in m:
            errors.append(f"missing {k}")
    if "images" not in m or "id" not in m:
        raise SystemExit(f"{path.name}: " + "; ".join(errors[:10]))
    names = set(coco80)
    if m.get("scored_labels") and not all(l in names or is_wildlife(l) for l in m["scored_labels"]):
        errors.append(f"scored_labels outside COCO-80 + wildlife: {sorted(set(m['scored_labels']) - names)}")
    if m.get("transform") not in (None, "grayscale"):
        errors.append(f"unknown transform {m['transform']!r}")
    if m.get("images_from"):
        src = OUT_DIR / f"{m['images_from']}.json"
        if not src.exists():
            errors.append(f"images_from {m['images_from']!r}: {src.name} does not exist")
        elif json.loads(src.read_text(encoding="utf-8")).get("images_from"):
            errors.append(f"images_from {m['images_from']!r} borrows its images too")
        if m["images"]:
            errors.append("a set with images_from inherits its images; keep `images` empty")
    elif not m["images"]:
        errors.append("no images")
    files = set()
    image_keys = {"file", "url", "sha256", "size", "width", "height", "tags", "objects"}
    object_keys = ({"label", "bbox", "ignore"}, {"label", "bbox", "ignore", "crowd"})
    for im in m["images"]:
        if set(im) != image_keys:
            errors.append(f"image keys {sorted(im)} != {sorted(image_keys)}")
            continue
        if im["file"] in files:
            errors.append(f"duplicate {im['file']}")
        files.add(im["file"])
        if not re.fullmatch(r"[0-9a-f]{64}", im["sha256"]):
            errors.append(f"{im['file']}: sha256 is not 64 lowercase hex digits")
        for o in im["objects"]:
            if set(o) not in object_keys:
                errors.append(f"{im['file']}: object keys {sorted(o)}")
                continue
            if o["label"] not in names and not is_wildlife(o["label"]):
                errors.append(f"{im['file']}: label {o['label']!r} is not a COCO-80 name or wildlife")
            x0, y0, x1, y1 = o["bbox"]
            if not (0 <= x0 < x1 <= im["width"] and 0 <= y0 < y1 <= im["height"]):
                errors.append(f"{im['file']}: bbox {o['bbox']} outside {im['width']}x{im['height']}")
        if work is not None:
            p = work / m["id"] / "images" / im["file"]
            if p.exists():
                data = p.read_bytes()
                if hashlib.sha256(data).hexdigest() != im["sha256"] or len(data) != im["size"]:
                    errors.append(f"{im['file']}: local copy sha/size mismatch")
        if head:
            try:
                status, size = http_head(im["url"])
                if status != 200 or (size is not None and size != im["size"]):
                    errors.append(f"{im['file']}: HEAD {status} size {size} != {im['size']}")
            except Exception as e:  # noqa: BLE001
                errors.append(f"{im['file']}: HEAD failed {e}")
    if errors:
        raise SystemExit(f"{path.name}: " + "; ".join(errors[:10]))
    return m


def stats(m: dict) -> str:
    scored = set(m.get("scored_labels") or [])
    tags = Counter(t for im in m["images"] for t in im["tags"])
    lab = Counter(o["label"] for im in m["images"] for o in im["objects"] if not o["ignore"]
                  and (not scored or o["label"] in scored))
    ign = sum(1 for im in m["images"] for o in im["objects"] if o["ignore"] or (scored and o["label"] not in scored))
    areas = sorted((o["bbox"][2] - o["bbox"][0]) * (o["bbox"][3] - o["bbox"][1]) for im in m["images"]
                   for o in im["objects"] if not o["ignore"] and (not scored or o["label"] in scored))
    small = sum(1 for a in areas if a < 32 ** 2)
    medium = sum(1 for a in areas if 32 ** 2 <= a < 96 ** 2)
    large = len(areas) - small - medium
    total = sum(im["size"] for im in m["images"])
    dims = Counter(f"{im['width']}x{im['height']}" for im in m["images"])
    lines = [
        f"== {m['id']}: {len(m['images'])} images, {total / 1e6:.1f} MB",
        f"   tags: {dict(sorted(tags.items()))}",
        f"   scored objects: {sum(lab.values())} {dict(lab.most_common())}; ignored/unscored: {ign}",
        f"   object sizes (COCO area bins): small {small}, medium {medium}, large {large}",
        f"   image dims (top): {dims.most_common(5)}",
    ]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True, help="download/cache directory (untrusted data)")
    ap.add_argument("--set", choices=["all", *DEFAULTS], default="all")
    ap.add_argument("--seed", type=int, default=20261010)
    ap.add_argument("--count", type=int, help="target image count (default per set)")
    ap.add_argument("--max-mb", type=float, help="byte budget in MB (default per set)")
    ap.add_argument("--validate", action="store_true", help="only validate existing manifests")
    ap.add_argument("--head", action="store_true", help="with --validate: HEAD every URL")
    ap.add_argument("--retag-grayscale", metavar="SET",
                    help="recompute the grayscale tag of an existing manifest from its pinned images")
    ap.add_argument("--review-dir", type=Path, help="nir-night: write the review contact sheet here")
    ap.add_argument("--exclude", type=Path, help="nir-night: file names to leave out (one per line)")
    ap.add_argument("--nacti-persons", action="store_true",
                    help="nir-night: include NACTI frames with person boxes (person becomes a scored positive)")
    args = ap.parse_args()
    coco80 = load_coco80()
    work = args.work.resolve()
    if args.retag_grayscale:
        write_manifest(retag_grayscale(args.retag_grayscale, work))
        print(stats(validate(OUT_DIR / f"{args.retag_grayscale}.json", coco80, work, False)))
        return
    sets = list(DEFAULTS) if args.set == "all" else [args.set]
    builders = {"bmd45-cctv": build_bmd45, "exdark-night": build_exdark, "coco-cctv": build_coco,
                "exdark-ir": build_exdark_ir, "nir-night": build_nir_night}
    for s in sets:
        if not args.validate:
            log(f"building {s}")
            ns = argparse.Namespace(**vars(args))
            ns.count = args.count or DEFAULTS[s]["count"]
            ns.max_mb = args.max_mb or DEFAULTS[s]["max_mb"]
            write_manifest(builders[s](work, coco80, ns))
        m = validate(OUT_DIR / f"{s}.json", coco80, work, args.head)
        if m.get("images_from"):
            print(f"== {m['id']}: images of {m['images_from']}, transform {m.get('transform')}")
            continue
        print(stats(m))


if __name__ == "__main__":
    main()
