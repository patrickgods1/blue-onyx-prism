#!/usr/bin/env python3
"""Keep the pinned dependencies honest: in step with each other, audited, and compared with upstream.

Usage (from anywhere; paths are relative to the repository root):
  check_deps.py consistency [--today YYYY-MM-DD]   offline, CI on every push (exit 1 on a mismatch)
  check_deps.py upstream [--out FILE]              online, weekly: pins with a newer upstream release
  check_deps.py audit [--out FILE]                 online, weekly: known vulnerabilities in the Python locks (OSV)
  check_deps.py --self-test

Nothing here changes a pin. Pins (URLs, SHA-256, sizes) change only in reviewed commits:
src/resources/catalog/ for downloads, scripts/requirements-model-*.txt plus the regenerated
scripts/model-export/locks/ for the Python environments, rust-toolchain.toml for Rust.

consistency checks:
- every `==` pin of requirements-model-export.txt is in each export lock at that version, and every pin of
  requirements-model-train.txt in each train lock (a local suffix such as `+cu126` is ignored);
- ULTRALYTICS_VERSION and TRANSFORMERS_VERSION in the catalog equal those pins;
- every workflow `toolchain:` equals rust-toolchain.toml, and Cargo.toml `rust-version` is its major.minor;
- the Dockerfile's OpenVINO runtime image tag equals OPENVINO_VERSION;
- no deny.toml advisory ignore is past its "Review by" date;
- a lock cutoff (`EXCLUDE_NEWER` in the gen scripts) older than STALE_LOCK_DAYS is a warning, not an error.

upstream and audit exit 0 with nothing to report, 10 when the report lists something (the workflow then
opens or updates an issue), 2 when a source could not be read at all.
"""
import argparse
import datetime as dt
import json
import os
import re
import sys
import tempfile
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
REPORT = 10
UNREADABLE = 2
STALE_LOCK_DAYS = 180
USER_AGENT = "blue-onyx-prism-ci"
SHOWN_IDS = 3

EXPORT_REQ = "scripts/requirements-model-export.txt"
TRAIN_REQ = "scripts/requirements-model-train.txt"
LOCKS = "scripts/model-export/locks"
TEST_REQ = "scripts/ci/requirements-test.txt"
CATALOG = "src/resources/catalog/mod.rs"
CATALOG_MODELS = "src/resources/catalog/models.rs"
CATALOG_TOOLS = "src/resources/catalog/tools.rs"
GEN_SCRIPTS = ("scripts/model-export/gen_locks.sh", "scripts/model-export/gen_train_locks.sh")

# Pins held back on purpose: shown in the upstream report with the reason, never counted as "newer".
HOLDS = {
    "ONNXRUNTIME_VERSION": "1.24.x is the last ONNX Runtime release with a DirectML package (cpu/directml/coreml flavors)",
    "CUDNN_VERSION": "must stay below 9.11 while the CUDA 12 lane covers Pascal (a catalog test checks it)",
}


# ----------------------------------------------------------------------------------------------
# Parsing
# ----------------------------------------------------------------------------------------------

def rust_const(text, name):
    """The value of `pub const NAME: &str = "...";` in Rust source, or None."""
    m = re.search(r'\bconst\s+' + re.escape(name) + r'\s*:\s*&str\s*=\s*"([^"]*)"', text)
    return m.group(1) if m else None


def norm_name(name):
    """PEP 503 name, without extras."""
    return re.sub(r"[-_.]+", "-", name.split("[", 1)[0]).lower()


def strip_local(version):
    return version.split("+", 1)[0]


def requirement_pins(text):
    """{normalized name: version} of `name==version` lines (comments, hashes, options skipped)."""
    out = {}
    for line in text.splitlines():
        line = line.split("#", 1)[0].strip().rstrip("\\").strip()
        m = re.match(r"^([A-Za-z0-9][A-Za-z0-9._\-]*(?:\[[^\]]*\])?)\s*==\s*([^\s;]+)", line)
        if m:
            out[norm_name(m.group(1))] = m.group(2)
    return out


def version_key(v):
    """Comparable key of a release version; None for a pre-release or something unparsable."""
    v = strip_local(v).lstrip("vV")
    if not re.fullmatch(r"\d+(\.\d+)*", v):
        return None
    parts = [int(p) for p in v.split(".")]
    while parts and parts[-1] == 0:
        parts.pop()
    return tuple(parts)


def is_newer(latest, pinned):
    a, b = version_key(latest), version_key(pinned)
    return a is not None and b is not None and a > b


def review_dates(deny_text):
    """[(advisory id, date)] of `Review by YYYY-MM-DD` in deny.toml ignores."""
    out = []
    for m in re.finditer(r'id\s*=\s*"([^"]+)"[^}]*?Review by (\d{4}-\d{2}-\d{2})', deny_text):
        out.append((m.group(1), dt.date.fromisoformat(m.group(2))))
    return out


def workflow_toolchains(root):
    out = []
    for wf in sorted((root / ".github/workflows").glob("*.yml")):
        for i, line in enumerate(wf.read_text(encoding="utf-8").splitlines(), 1):
            m = re.match(r"^\s*toolchain:\s*['\"]?([^\s'\"#]+)", line)
            if m:
                out.append((f"{wf.relative_to(root).as_posix()}:{i}", m.group(1)))
    return out


def read(root, rel):
    return (root / rel).read_text(encoding="utf-8")


# ----------------------------------------------------------------------------------------------
# consistency
# ----------------------------------------------------------------------------------------------

def consistency(root, today):
    errors, warnings = [], []

    def locks_match(req_rel, lock_glob_include, lock_glob_exclude=None):
        pins = requirement_pins(read(root, req_rel))
        locks = [p for p in sorted((root / LOCKS).glob("*.txt"))
                 if lock_glob_include(p.name) and not (lock_glob_exclude and lock_glob_exclude(p.name))]
        if not locks:
            errors.append(f"{req_rel}: no lock file found under {LOCKS}")
        for lock in locks:
            have = requirement_pins(lock.read_text(encoding="utf-8"))
            for name, ver in pins.items():
                got = have.get(name)
                if got is None:
                    errors.append(f"{lock.relative_to(root).as_posix()}: {name} missing (pinned {ver} in {req_rel}); regenerate the locks")
                elif strip_local(got) != ver:
                    errors.append(f"{lock.relative_to(root).as_posix()}: {name}=={got}, but {req_rel} pins {ver}; regenerate the locks")
        return pins

    export_pins = locks_match(EXPORT_REQ, lambda n: not n.startswith("train-"))
    locks_match(TRAIN_REQ, lambda n: n.startswith("train-"))

    models = read(root, CATALOG_MODELS)
    for const, pkg in (("ULTRALYTICS_VERSION", "ultralytics"), ("TRANSFORMERS_VERSION", "transformers")):
        v = rust_const(models, const)
        if v is None:
            errors.append(f"{CATALOG_MODELS}: {const} not found (renamed? update scripts/ci/check_deps.py)")
        elif export_pins.get(pkg) != v:
            errors.append(f"{CATALOG_MODELS}: {const} = {v}, but {EXPORT_REQ} pins {pkg}=={export_pins.get(pkg)}")

    channel = re.search(r'channel\s*=\s*"([^"]+)"', read(root, "rust-toolchain.toml"))
    channel = channel.group(1) if channel else None
    if channel is None:
        errors.append("rust-toolchain.toml: no channel")
    else:
        for where, tc in workflow_toolchains(root):
            if tc != channel:
                errors.append(f"{where}: toolchain {tc}, but rust-toolchain.toml pins {channel}")
        rv = re.search(r'^rust-version\s*=\s*"([^"]+)"', read(root, "Cargo.toml"), re.M)
        if rv is None or ".".join(channel.split(".")[:2]) != rv.group(1):
            errors.append(f"Cargo.toml: rust-version {rv.group(1) if rv else '(none)'} is not the major.minor of rust-toolchain.toml {channel}")

    ov = rust_const(read(root, CATALOG), "OPENVINO_VERSION")
    tag = re.search(r"openvino/ubuntu24_runtime:([^@\s]+)", read(root, "Dockerfile"))
    if ov is None:
        errors.append(f"{CATALOG}: OPENVINO_VERSION not found")
    elif tag is None:
        errors.append("Dockerfile: no openvino/ubuntu24_runtime image")
    elif tag.group(1) != ov:
        errors.append(f"Dockerfile: OpenVINO runtime image {tag.group(1)}, but the catalog pins OPENVINO_VERSION {ov}; bump the image (tag and digest) with the catalog")

    for adv, date in review_dates(read(root, "deny.toml")):
        if date < today:
            errors.append(f"deny.toml: the ignore of {adv} was due for review on {date}: check for a fixed release or a replacement, then remove the ignore or move the date")

    for rel in GEN_SCRIPTS:
        m = re.search(r'EXCLUDE_NEWER="\$\{2:-(\d{4}-\d{2}-\d{2})', read(root, rel))
        if m is None:
            warnings.append(f"{rel}: no default EXCLUDE_NEWER found")
        else:
            age = (today - dt.date.fromisoformat(m.group(1))).days
            if age > STALE_LOCK_DAYS:
                warnings.append(f"{rel}: the lock cutoff {m.group(1)} is {age} days old; consider regenerating the locks")
    return errors, warnings


# ----------------------------------------------------------------------------------------------
# Online sources
# ----------------------------------------------------------------------------------------------

def fetch_json(url, data=None):
    headers = {"User-Agent": USER_AGENT, "Accept": "application/json"}
    if data is not None:
        headers["Content-Type"] = "application/json"
        data = json.dumps(data).encode()
    token = os.environ.get("GITHUB_TOKEN") or os.environ.get("GH_TOKEN")
    if token and url.startswith("https://api.github.com/"):
        headers["Authorization"] = f"Bearer {token}"
    req = urllib.request.Request(url, data=data, headers=headers)
    with urllib.request.urlopen(req, timeout=30) as r:
        return json.load(r)


def github_latest(repo):
    return fetch_json(f"https://api.github.com/repos/{repo}/releases/latest")["tag_name"].lstrip("v")


def pypi_latest(name):
    return fetch_json(f"https://pypi.org/pypi/{name}/json")["info"]["version"]


def nuget_latest(name):
    versions = fetch_json(f"https://api.nuget.org/v3-flatcontainer/{name.lower()}/index.json")["versions"]
    releases = [v for v in versions if version_key(v) is not None]
    return max(releases, key=version_key) if releases else versions[-1]


def catalog_sources(root):
    """[(component, const name, pinned, fetch latest)] for the runtime pins in the catalog."""
    cat, tools = read(root, CATALOG), read(root, CATALOG_TOOLS)
    rows = [
        ("OpenVINO runtime", "OPENVINO_VERSION", rust_const(cat, "OPENVINO_VERSION"),
         lambda: github_latest("openvinotoolkit/openvino")),
        ("ONNX Runtime (cpu/directml/coreml)", "ONNXRUNTIME_VERSION", rust_const(cat, "ONNXRUNTIME_VERSION"),
         lambda: github_latest("microsoft/onnxruntime")),
        ("ONNX Runtime (cuda)", "ONNXRUNTIME_CUDA_VERSION", rust_const(cat, "ONNXRUNTIME_CUDA_VERSION"),
         lambda: github_latest("microsoft/onnxruntime")),
        ("DirectML", "DIRECTML_VERSION", rust_const(cat, "DIRECTML_VERSION"),
         lambda: nuget_latest("Microsoft.AI.DirectML")),
        ("cuDNN (nvidia-cudnn-cu12)", "CUDNN_VERSION", rust_const(cat, "CUDNN_VERSION"),
         lambda: pypi_latest("nvidia-cudnn-cu12")),
        ("uv", "UV_VERSION", rust_const(cat, "UV_VERSION"), lambda: github_latest("astral-sh/uv")),
        ("VC++ runtime (msvc-runtime)", "VC_RUNTIME_VERSION", rust_const(tools, "VC_RUNTIME_VERSION"),
         lambda: pypi_latest("msvc-runtime")),
    ]
    return rows


def upstream(root):
    rows, failures = [], 0
    for component, const, pinned, latest_fn in catalog_sources(root):
        if pinned is None:
            rows.append((component, "?", "?", f"{const} not found in the catalog"))
            failures += 1
            continue
        try:
            latest = latest_fn()
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            rows.append((component, pinned, "?", f"could not read upstream: {e}"))
            failures += 1
            continue
        rows.append((component, pinned, latest, HOLDS.get(const, "")))
    pins = requirement_pins(read(root, EXPORT_REQ))
    pins.update(requirement_pins(read(root, TRAIN_REQ)))
    for name, pinned in sorted(pins.items()):
        try:
            latest = pypi_latest(name)
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            rows.append((f"{name} (Python)", pinned, "?", f"could not read PyPI: {e}"))
            failures += 1
            continue
        rows.append((f"{name} (Python)", pinned, latest, ""))
    total = len(rows)
    newer = [r for r in rows if r[2] != "?" and is_newer(r[2], r[1]) and not any(r[3] == h for h in HOLDS.values())]
    held = [r for r in rows if r[2] != "?" and is_newer(r[2], r[1]) and any(r[3] == h for h in HOLDS.values())]
    lines = []
    if newer:
        lines += ["Newer upstream releases than the pinned versions. Nothing was changed: bump a pin in a reviewed",
                  "commit (catalog URL, size and SHA-256; or the requirements pin and regenerated locks), and read the",
                  "upstream release notes for breaking changes first.", "",
                  "| Component | Pinned | Latest |", "|---|---|---|"]
        lines += [f"| {c} | {p} | {l} |" for c, p, l, _ in newer]
    if held:
        lines += ["", "Held back on purpose:", "", "| Component | Pinned | Latest | Why |", "|---|---|---|---|"]
        lines += [f"| {c} | {p} | {l} | {n} |" for c, p, l, n in held]
    errs = [r for r in rows if r[2] == "?"]
    if errs:
        lines += ["", "Could not check:", ""] + [f"- {c}: {n}" for c, _, _, n in errs]
    if not lines:
        lines = ["Every pin is at its latest upstream release."]
    code = UNREADABLE if failures == total else (REPORT if newer else 0)
    return "\n".join(lines) + "\n", code


def lock_packages(root):
    """{(name, version): [files]} over every Python lock the project installs (local suffix dropped)."""
    out = {}
    files = sorted((root / LOCKS).glob("*.txt")) + [root / TEST_REQ]
    for f in files:
        for name, ver in requirement_pins(f.read_text(encoding="utf-8")).items():
            out.setdefault((name, strip_local(ver)), []).append(f.name)
    return out


def audit(root):
    where = lock_packages(root)
    pkgs = sorted(where)
    found = {}
    for i in range(0, len(pkgs), 500):
        chunk = pkgs[i:i + 500]
        try:
            ans = fetch_json("https://api.osv.dev/v1/querybatch", {
                "queries": [{"package": {"name": n, "ecosystem": "PyPI"}, "version": v} for n, v in chunk]})
        except (urllib.error.URLError, OSError, ValueError) as e:
            return f"Could not query OSV: {e}\n", UNREADABLE
        for (n, v), res in zip(chunk, ans.get("results", [])):
            ids = sorted({x["id"] for x in res.get("vulns", [])})
            if ids:
                found[(n, v)] = ids
    if not found:
        return f"No known vulnerabilities in the {len(pkgs)} locked Python packages (OSV).\n", 0
    lines = [f"Known vulnerabilities in locked Python packages ({len(found)} of {len(pkgs)}, from OSV).",
             "Bump the pin in scripts/requirements-model-*.txt (or a transitive one by regenerating the locks with",
             "a later --exclude-newer) in a reviewed commit, after checking the advisory applies.", "",
             "| Package | Version | Advisories | Locked in |", "|---|---|---|---|"]
    for (n, v), ids in sorted(found.items()):
        shown = ", ".join(f"[{x}](https://osv.dev/vulnerability/{x})" for x in ids[:SHOWN_IDS])
        more = f" and {len(ids) - SHOWN_IDS} more" if len(ids) > SHOWN_IDS else ""
        files = ", ".join(where[(n, v)])
        lines.append(f"| {n} | {v} | {len(ids)}: {shown}{more} | {files} |")
    return "\n".join(lines) + "\n", REPORT


# ----------------------------------------------------------------------------------------------
# Self-test
# ----------------------------------------------------------------------------------------------

def self_test():
    assert version_key("v1.31.0") == (1, 31)
    assert version_key("2026.4.0") == version_key("2026.4")
    assert version_key("1.0.0rc1") is None
    assert is_newer("1.31.1", "1.31.0") and not is_newer("1.31.0", "1.31.0")
    assert is_newer("2.14.1", "2.14.0+cu126")
    assert not is_newer("2.15.0rc1", "2.14.0")
    assert norm_name("torchmetrics[detection]") == "torchmetrics"
    assert norm_name("Huggingface_Hub") == "huggingface-hub"
    pins = requirement_pins("# c\nfoo==1.0 \\\n    --hash=sha256:ab\nBar_baz[x]==2.0+cpu ; sys_platform=='x'\n--index-url x\nqux>=1\n")
    assert pins == {"foo": "1.0", "bar-baz": "2.0+cpu"}, pins
    assert rust_const('pub const A_VERSION: &str = "1.2.3"; // x', "A_VERSION") == "1.2.3"
    assert rust_const('pub const AB_VERSION: &str = "9";', "B_VERSION") is None
    dates = review_dates('ignore = [\n { id = "RUSTSEC-1", reason = "x. Review by 2027-04-01" },\n]')
    assert dates == [("RUSTSEC-1", dt.date(2027, 4, 1))], dates

    with tempfile.TemporaryDirectory() as d:
        root = Path(d)

        def put(rel, text):
            p = root / rel
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(text, encoding="utf-8")

        put(EXPORT_REQ, "ultralytics==8.4.163\ntorch==2.14.0\ntransformers==5.17.0\n")
        put(TRAIN_REQ, "ultralytics==8.4.163\ntorch==2.14.0\ntransformers==5.17.0\nyt-dlp==2026.8.19\n")
        lock = "ultralytics==8.4.163 \\\n    --hash=sha256:aa\ntorch==2.14.0+cpu \\\n    --hash=sha256:bb\ntransformers==5.17.0\n"
        put(f"{LOCKS}/macos-arm64.txt", lock)
        put(f"{LOCKS}/train-macos-arm64.txt", lock + "yt-dlp==2026.8.19\n")
        put(CATALOG_MODELS, 'pub const ULTRALYTICS_VERSION: &str = "8.4.163";\npub const TRANSFORMERS_VERSION: &str = "5.17.0";\n')
        put(CATALOG, 'pub const OPENVINO_VERSION: &str = "2026.4.0";\n')
        put("rust-toolchain.toml", '[toolchain]\nchannel = "1.99.0"\n')
        put("Cargo.toml", '[package]\nrust-version = "1.99"\n')
        put(".github/workflows/ci.yml", "steps:\n  - with:\n      toolchain: 1.99.0 # keep equal\n")
        put("Dockerfile", "FROM openvino/ubuntu24_runtime:2026.4.0@sha256:00\n")
        put("deny.toml", 'ignore = [ { id = "RUSTSEC-1", reason = "Review by 2027-04-01" } ]\n')
        for rel in GEN_SCRIPTS:
            put(rel, 'EXCLUDE_NEWER="${2:-2026-09-26T00:00:00Z}"\n')

        today = dt.date(2026, 11, 7)
        errors, warnings = consistency(root, today)
        assert errors == [] and warnings == [], (errors, warnings)

        errors, warnings = consistency(root, dt.date(2027, 4, 2))
        assert len(errors) == 1 and "RUSTSEC-1" in errors[0], errors
        assert len(warnings) == 2 and "days old" in warnings[0], warnings

        put(f"{LOCKS}/train-macos-arm64.txt", lock + "yt-dlp==2026.7.1\n")
        put(".github/workflows/release.yml", "      toolchain: 1.98.0\n")
        put("Dockerfile", "FROM openvino/ubuntu24_runtime:2026.3.0@sha256:00\n")
        put(CATALOG_MODELS, 'pub const ULTRALYTICS_VERSION: &str = "8.4.162";\npub const TRANSFORMERS_VERSION: &str = "5.17.0";\n')
        errors, _ = consistency(root, today)
        joined = "\n".join(errors)
        for needle in ("yt-dlp==2026.7.1", "release.yml:1: toolchain 1.98.0", "image 2026.3.0", "ULTRALYTICS_VERSION = 8.4.162"):
            assert needle in joined, (needle, errors)
        assert len(errors) == 4, errors

        put(f"{TEST_REQ}", "numpy==2.4.6 \\\n    --hash=sha256:cc\n")
        pk = lock_packages(root)
        assert pk[("torch", "2.14.0")] == ["macos-arm64.txt", "train-macos-arm64.txt"], pk
        assert pk[("numpy", "2.4.6")] == ["requirements-test.txt"], pk
    print("check_deps self-test ok")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("command", nargs="?", choices=("consistency", "upstream", "audit"))
    ap.add_argument("--self-test", action="store_true")
    ap.add_argument("--today", type=dt.date.fromisoformat, default=None)
    ap.add_argument("--out", type=Path, help="also write the report (Markdown) here")
    a = ap.parse_args()
    if a.self_test:
        self_test()
        return 0
    if a.command is None:
        ap.error("a command or --self-test is required")
    if a.command == "consistency":
        errors, warnings = consistency(ROOT, a.today or dt.date.today())
        for w in warnings:
            print(f"warning: {w}")
        for e in errors:
            print(f"error: {e}")
        if not errors:
            print("dependency pins consistent")
        return 1 if errors else 0
    text, code = upstream(ROOT) if a.command == "upstream" else audit(ROOT)
    sys.stdout.write(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
