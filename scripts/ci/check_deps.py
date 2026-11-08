#!/usr/bin/env python3
"""Keep the pinned dependencies honest: in step with each other, audited, and compared with upstream.

Usage (from anywhere; paths are relative to the repository root):
  check_deps.py consistency [--today YYYY-MM-DD]   offline, CI on every push (exit 1 on a mismatch)
  check_deps.py upstream [--out FILE] [--today D]  online, weekly: pins with a newer upstream release that is due
  check_deps.py audit [--out FILE]                 online, weekly: known vulnerabilities in the Python locks (OSV)
  check_deps.py conflicts                          online: a Python pin outside what another pin declares (exit 1)
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

upstream calls a release due when it is newer than the pin, at least MIN_AGE_DAYS old and (Python)
runs on the environments' Python (Requires-Python) and (Python) fits what the other pins declare
(Requires-Dist, with the extras we install: USED_EXTRAS); newer releases that are not due yet are listed apart,
with the reason, and do not keep the issue open. Pins held back on purpose (HOLDS) are listed apart too.

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
# A pin moves only to a release at least this old (docs/RELEASE_CHECKLIST.md).
MIN_AGE_DAYS = 14
# Python of the export / train environments (src/resources/export/mod.rs PYTHON_VERSION; the newest 3.11).
PYTHON = (3, 11, 99)

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
    "VC_RUNTIME_VERSION": "14.5x is the Visual Studio 2026 runtime (no cp311 wheel, the 2026 redistribution terms, "
                          "vcomp140 regressions reported on 14.50): bump after a check on a real Windows machine",
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


def release_rows(versions):
    """[(version, first upload date, requires_python)] -> releases only (no pre-release), oldest first."""
    rows = [r for r in versions if version_key(r[0]) is not None and r[1] is not None]
    return sorted(rows, key=lambda r: version_key(r[0]))


def github_releases(repo):
    """[(version, published date, None)] of a GitHub repository's releases (drafts and pre-releases
    skipped; tags that are not a plain version, such as onnxruntime's `plugin-ep-cuda/v0.3.0`, too)."""
    rels = fetch_json(f"https://api.github.com/repos/{repo}/releases?per_page=100")
    out = []
    for r in rels:
        tag = r.get("tag_name", "")
        if r.get("draft") or r.get("prerelease") or not re.fullmatch(r"v?\d+(\.\d+)*", tag):
            continue
        out.append((tag.lstrip("v"), dt.date.fromisoformat(r["published_at"][:10]), None))
    return out


def pypi_releases(name):
    """[(version, first upload date, requires_python)] of a PyPI project."""
    out = []
    for ver, files in fetch_json(f"https://pypi.org/pypi/{name}/json")["releases"].items():
        files = [f for f in files if not f.get("yanked")]
        if not files:
            continue
        first = min(f["upload_time"][:10] for f in files)
        out.append((ver, dt.date.fromisoformat(first), files[0].get("requires_python")))
    return out


def nuget_releases(name):
    """[(version, published date, None)] of a NuGet package (registration index, pages fetched when
    they are not inlined)."""
    index = fetch_json(f"https://api.nuget.org/v3/registration5-semver1/{name.lower()}/index.json")
    out = []
    for page in index.get("items", []):
        items = page.get("items")
        if items is None:
            items = fetch_json(page["@id"]).get("items", [])
        for it in items:
            e = it["catalogEntry"]
            if e.get("listed", True):
                out.append((e["version"], dt.date.fromisoformat(e["published"][:10]), None))
    return out


def python_ok(spec, py=PYTHON):
    """Whether a Requires-Python specifier (`>=3.10`, `>=3.9,<3.14`, `!=3.0.*`, `~=3.8`) admits the
    Python of the environments; a missing one admits it."""
    return spec_ok(spec, py)


def spec_ok(spec, version):
    """Whether a version specifier (`<1.9.0,>=1.8.2`, `==1.0.1`, `!=3.0.*`, `~=3.8`) admits `version`
    (a tuple of ints, or a release string); an empty one admits everything, clauses that are not
    plain release versions are skipped."""
    if not spec:
        return True
    if isinstance(version, str):
        version = tuple(int(x) for x in strip_local(version).split("."))
    have = list(version)
    for clause in spec.split(","):
        m = re.fullmatch(r"\s*(~=|===|==|!=|<=|>=|<|>)\s*([0-9.]+)(\.\*)?\s*", clause)
        if m is None:
            continue
        op, ver, star = m.groups()
        want = [int(x) for x in ver.split(".")]
        if star:
            same = have[:len(want)] == want
            if (op == "==" and not same) or (op == "!=" and same):
                return False
            continue
        n = max(len(want), len(have))
        a, b = have + [0] * (n - len(have)), want + [0] * (n - len(want))
        ok = {"==": a == b, "===": a == b, "!=": a != b, "<": a < b, "<=": a <= b, ">": a > b,
              ">=": a >= b, "~=": a >= b and a[:len(want) - 1] == b[:len(want) - 1]}[op]
        if not ok:
            return False
    return True


# Extras installed of a pinned package (requirements-model-train.txt lists rfdetr's `train` extra by
# hand, minus roboflow); requirements under any other extra of it are not ours.
USED_EXTRAS = {"rfdetr": {"train"}, "torchmetrics": {"detection"}}


def parse_requirement(line):
    """(name, specifier, extras of its marker) of a Requires-Dist line, or None."""
    req, _, marker = line.partition(";")
    m = re.match(r"\s*([A-Za-z0-9][A-Za-z0-9._\-]*)\s*(?:\[[^\]]*\])?\s*\(?([^)]*)\)?\s*$", req)
    if m is None:
        return None
    extras = set(re.findall(r"""extra\s*==\s*['"]([^'"]+)['"]""", marker))
    return norm_name(m.group(1)), m.group(2).replace(" ", ""), extras


def pin_constraints(pins, requires_dist):
    """{dependency: [(specifier, "pkg ver")]}: what each pinned package (`requires_dist(name, ver)`)
    declares about the others, counting only its requirements without an extra or under an extra we
    install (USED_EXTRAS)."""
    out = {}
    for name, ver in pins.items():
        for line in requires_dist(name, ver) or []:
            parsed = parse_requirement(line)
            if parsed is None:
                continue
            dep, spec, extras = parsed
            if extras and not extras & USED_EXTRAS.get(name, set()):
                continue
            if dep in pins and dep != name and spec:
                out.setdefault(dep, []).append((spec, f"{name} {ver}"))
    return out


def pin_conflicts(pins, constraints):
    """["torchmetrics==1.9.0: rfdetr 1.11.2 requires <1.9.0,>=1.8.2", ...]: pins outside what another
    pin declares (`pin_constraints`)."""
    out = []
    for name, ver in sorted(pins.items()):
        for spec, who in constraints.get(name, ()):
            if version_key(ver) is not None and not spec_ok(spec, ver):
                out.append(f"{name}=={ver}: {who} requires {spec}")
    return out


def python_pins(root):
    pins = requirement_pins(read(root, EXPORT_REQ))
    pins.update(requirement_pins(read(root, TRAIN_REQ)))
    return pins


def conflicts(root):
    """Online: the export and train pins against each other's Requires-Dist. uv cannot see these:
    the train requirements list rfdetr's `train` extra by hand (without roboflow), so its ranges for
    torchmetrics, hotcoco, vernier, ... are only declared, never resolved."""
    pins = python_pins(root)
    try:
        found = pin_conflicts(pins, pin_constraints(pins, pypi_requires_dist))
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        return f"Could not read PyPI: {e}\n", UNREADABLE
    if not found:
        return f"The {len(pins)} Python pins fit each other's declared requirements.\n", 0
    return "Pins outside what another pin requires:\n" + "".join(f"- {c}\n" for c in found), 1


def pypi_requires_dist(name, version):
    return fetch_json(f"https://pypi.org/pypi/{name}/{version}/json")["info"].get("requires_dist")


def due_release(rows, today, constraints=()):
    """(due, newest, why) over `release_rows`: `due` is the newest release from before the day
    MIN_AGE_DAYS ago (the lock cutoff `--exclude-newer <that day>T00:00:00Z` admits it) that runs on
    the environments' Python, the one a pin may move to; `newest` the newest release of all, `why` says
    why newest is not due (None when it is)."""
    rows = release_rows(rows)
    if not rows:
        return None, None, None
    cutoff = today - dt.timedelta(days=MIN_AGE_DAYS)
    fits = [r for r in rows if python_ok(r[2]) and all(spec_ok(s, r[0]) for s, _ in constraints)]
    old = [r for r in fits if r[1] < cutoff]
    due = old[-1][0] if old else None
    newest = rows[-1]
    why = None
    if due != newest[0]:
        blocked = [(s, who) for s, who in constraints if not spec_ok(s, newest[0])]
        if blocked:
            why = f"{newest[0]} is outside what {blocked[0][1]} requires ({blocked[0][0]})"
        elif not python_ok(newest[2]):
            why = f"{newest[0]} needs Python {newest[2]}; the environments run Python {'.'.join(map(str, PYTHON[:2]))}"
        else:
            why = f"{newest[0]} was released {newest[1]}, due from {newest[1] + dt.timedelta(days=MIN_AGE_DAYS + 1)}"
    return due, newest[0], why


def catalog_sources(root):
    """[(component, const name, pinned, fetch releases)] for the runtime pins in the catalog."""
    cat, tools = read(root, CATALOG), read(root, CATALOG_TOOLS)
    rows = [
        ("OpenVINO runtime", "OPENVINO_VERSION", rust_const(cat, "OPENVINO_VERSION"),
         lambda: github_releases("openvinotoolkit/openvino")),
        ("ONNX Runtime (cpu/directml/coreml)", "ONNXRUNTIME_VERSION", rust_const(cat, "ONNXRUNTIME_VERSION"),
         lambda: github_releases("microsoft/onnxruntime")),
        ("ONNX Runtime (cuda)", "ONNXRUNTIME_CUDA_VERSION", rust_const(cat, "ONNXRUNTIME_CUDA_VERSION"),
         lambda: github_releases("microsoft/onnxruntime")),
        ("DirectML", "DIRECTML_VERSION", rust_const(cat, "DIRECTML_VERSION"),
         lambda: nuget_releases("Microsoft.AI.DirectML")),
        ("cuDNN (nvidia-cudnn-cu12)", "CUDNN_VERSION", rust_const(cat, "CUDNN_VERSION"),
         lambda: [(v, d, None) for v, d, _ in pypi_releases("nvidia-cudnn-cu12")]),
        ("uv", "UV_VERSION", rust_const(cat, "UV_VERSION"), lambda: github_releases("astral-sh/uv")),
        ("VC++ runtime (msvc-runtime)", "VC_RUNTIME_VERSION", rust_const(tools, "VC_RUNTIME_VERSION"),
         lambda: [(v, d, None) for v, d, _ in pypi_releases("msvc-runtime")]),
    ]
    return rows


def upstream_report(rows):
    """Markdown report and whether anything is due, from rows of
    {component, pinned, due, newest, why, hold} (due/newest None = could not be read; `why` then
    says why)."""
    unread = [r for r in rows if r["newest"] is None]
    read_ok = [r for r in rows if r["newest"] is not None]
    due = [r for r in read_ok if r["due"] and is_newer(r["due"], r["pinned"]) and not r["hold"]]
    held = [r for r in read_ok if r["hold"] and is_newer(r["newest"], r["pinned"])]
    waiting = [r for r in read_ok if not r["hold"] and is_newer(r["newest"], r["pinned"])
               and r["why"] and not (r["due"] and is_newer(r["due"], r["pinned"]))]
    later = [r for r in due if r["why"]]
    lines = []
    if due:
        lines += ["Newer upstream releases than the pinned versions, at least "
                  f"{MIN_AGE_DAYS} days old. Nothing was changed: bump a pin in a reviewed",
                  "commit (catalog URL, size and SHA-256; or the requirements pin and regenerated locks), and read the",
                  "upstream release notes for breaking changes first.", "",
                  "| Component | Pinned | Due | Newest |", "|---|---|---|---|"]
        lines += [f"| {r['component']} | {r['pinned']} | {r['due']} | {r['newest']} |" for r in due]
        if later:
            lines += [""] + [f"- {r['component']}: {r['why']}" for r in later]
    if waiting:
        lines += ["", "Newer, but not due yet:", "", "| Component | Pinned | Newest | Why not yet |", "|---|---|---|---|"]
        lines += [f"| {r['component']} | {r['pinned']} | {r['newest']} | {r['why']} |" for r in waiting]
    if held:
        lines += ["", "Held back on purpose:", "", "| Component | Pinned | Newest | Why |", "|---|---|---|---|"]
        lines += [f"| {r['component']} | {r['pinned']} | {r['newest']} | {r['hold']} |" for r in held]
    if unread:
        lines += ["", "Could not check:", ""] + [f"- {r['component']}: {r['why']}" for r in unread]
    if not lines:
        lines = ["Every pin is at its latest upstream release."]
    elif not due:
        lines = ["No pin is due for a bump."] + lines
    return "\n".join(lines) + "\n", bool(due)


def upstream(root, today):
    rows, failures = [], 0

    def add(component, pinned, fetch, hold="", constraints=()):
        nonlocal failures
        try:
            d, newest, why = due_release(fetch(), today, constraints)
            if newest is None and version_key(pinned) is None:
                return  # a pre-release pin of a project with no release yet (torch-hungarian): by hand
            if newest is None:
                raise ValueError("no releases listed")
        except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
            failures += 1
            rows.append(dict(component=component, pinned=pinned, due=None, newest=None,
                             why=f"could not read upstream: {e}", hold=hold))
            return
        rows.append(dict(component=component, pinned=pinned, due=d, newest=newest, why=why, hold=hold))

    for component, const, pinned, fetch in catalog_sources(root):
        if pinned is None:
            failures += 1
            rows.append(dict(component=component, pinned="?", due=None, newest=None,
                             why=f"{const} not found in the catalog", hold=""))
            continue
        add(component, pinned, fetch, HOLDS.get(const, ""))
    pins = python_pins(root)
    try:
        cons = pin_constraints(pins, pypi_requires_dist)
        for c in pin_conflicts(pins, cons):
            rows.append(dict(component=c.split("==", 1)[0] + " (Python)", pinned=c.split("==", 1)[1].split(":", 1)[0],
                             due=None, newest=None, why=f"pin conflict: {c}", hold=""))
            failures += 1
    except (urllib.error.URLError, OSError, KeyError, ValueError) as e:
        cons = {}
        failures += 1
        rows.append(dict(component="Requirements of the pinned Python packages", pinned="?", due=None,
                         newest=None, why=f"could not read PyPI: {e}", hold=""))
    for name, pinned in sorted(pins.items()):
        add(f"{name} (Python)", pinned, lambda n=name: pypi_releases(n), HOLDS.get(name, ""),
            cons.get(name, ()))
    text, any_due = upstream_report(rows)
    code = UNREADABLE if failures == len(rows) else (REPORT if any_due else 0)
    return text, code


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

    for spec, ok in ((None, True), ("", True), (">=3.10", True), (">=3.12", False), (">=3.9,<3.14", True),
                     ("<3.11", False), ("!=3.11.*", False), ("==3.11.*", True), ("~=3.8", True), ("~=3.12", False),
                     (">=3.10.0", True), ("!=3.0.*,!=3.1.*,>=2.7", True)):
        assert python_ok(spec) is ok, (spec, ok)
    assert spec_ok("<1.9.0,>=1.8.2", "1.8.2") and not spec_ok("<1.9.0,>=1.8.2", "1.9.0")
    assert spec_ok("==1.0.1", "1.0.1") and not spec_ok("==1.0.1", "1.3.0") and spec_ok("", "9")
    assert spec_ok("==2.9.*", "2.9.1") and not spec_ok("==2.9.*", "2.14.1")
    assert parse_requirement('torchmetrics[detection]<1.9.0,>=1.8.2; extra == "train"') == (
        "torchmetrics", "<1.9.0,>=1.8.2", {"train"})
    assert parse_requirement("transformers (<6.0.0,>=5.1.0)") == ("transformers", "<6.0.0,>=5.1.0", set())
    dist = {("rfdetr", "1.11.2"): ['transformers<6.0.0,>=5.1.0', 'torchmetrics[detection]<1.9.0,>=1.8.2; extra == "train"',
                                   'hotcoco==1.0.1; extra == "train"', 'torch==2.9.*; sys_platform == "linux" and extra == "xla"',
                                   'roboflow; extra == "train"', 'peft; extra == "train"'],
            ("torchmetrics", "1.8.2"): ['torch>=2.0.0', 'scipy>1.0.0; extra == "detection"', 'numpy>1; extra == "image"']}
    cons = pin_constraints({"rfdetr": "1.11.2", "torchmetrics": "1.8.2", "hotcoco": "1.0.1", "torch": "2.14.1",
                            "transformers": "5.20.2", "scipy": "1.17.1", "peft": "0.21.2"},
                           lambda n, v: dist.get((n, v)))
    assert cons == {"transformers": [("<6.0.0,>=5.1.0", "rfdetr 1.11.2")],
                    "torchmetrics": [("<1.9.0,>=1.8.2", "rfdetr 1.11.2")],
                    "hotcoco": [("==1.0.1", "rfdetr 1.11.2")],
                    "torch": [(">=2.0.0", "torchmetrics 1.8.2")],
                    "scipy": [(">1.0.0", "torchmetrics 1.8.2")]}, cons
    day = dt.date(2026, 11, 7)
    d = dt.date.fromisoformat
    pins = {"rfdetr": "1.11.2", "torchmetrics": "1.9.0", "hotcoco": "1.0.1", "torch": "2.14.1"}
    assert pin_conflicts(pins, cons) == ["torchmetrics==1.9.0: rfdetr 1.11.2 requires <1.9.0,>=1.8.2"]
    pins["torchmetrics"] = "1.8.2"
    assert pin_conflicts(pins, cons) == []
    tm = [("1.8.2", d("2025-09-01"), None), ("1.9.0", d("2026-03-09"), None)]
    assert due_release(tm, day, cons["torchmetrics"]) == (
        "1.8.2", "1.9.0", "1.9.0 is outside what rfdetr 1.11.2 requires (<1.9.0,>=1.8.2)")
    rel = [("1.17.1", d("2026-02-01"), ">=3.11"), ("1.18.1", d("2026-08-21"), ">=3.12"),
           ("2.0.0rc1", d("2026-01-01"), None), ("1.17.0", d("2026-01-01"), ">=3.11")]
    assert due_release(rel, day) == ("1.17.1", "1.18.1", "1.18.1 needs Python >=3.12; the environments run Python 3.11")
    rel = [("8.4.163", d("2026-09-26"), None), ("8.4.184", d("2026-10-22"), None), ("8.4.188", d("2026-10-24"), None),
           ("8.4.207", d("2026-11-07"), None)]
    assert due_release(rel, day) == ("8.4.184", "8.4.207", "8.4.207 was released 2026-11-07, due from 2026-11-22")
    assert due_release(rel[:2], day) == ("8.4.184", "8.4.184", None)
    assert due_release(rel[:3], dt.date(2026, 11, 8))[0] == "8.4.188"
    assert due_release([("1.0", d("2026-11-01"), None)], day) == (None, "1.0", "1.0 was released 2026-11-01, due from 2026-11-16")
    assert due_release([], day) == (None, None, None)

    def row(c, p, due, newest, why=None, hold=""):
        return dict(component=c, pinned=p, due=due, newest=newest, why=why, hold=hold)
    text, any_due = upstream_report([
        row("ultralytics (Python)", "8.4.163", "8.4.184", "8.4.207", "8.4.207 was released 2026-11-07, due from 2026-11-22"),
        row("rfdetr (Python)", "1.11.2", "1.11.2", "1.12.0", "1.12.0 was released 2026-11-06, due from 2026-11-21"),
        row("scipy (Python)", "1.17.1", "1.17.1", "1.18.1", "1.18.1 needs Python >=3.12; the environments run Python 3.11"),
        row("ONNX Runtime (cpu/directml/coreml)", "1.24.4", "1.31.0", "1.31.1", hold="DirectML"),
        row("onnx (Python)", "1.23.2", "1.23.2", "1.23.2"),
        row("uv", "0.13.5", None, None, "could not read upstream: timeout")])
    assert any_due
    assert "| ultralytics (Python) | 8.4.163 | 8.4.184 | 8.4.207 |" in text, text
    assert "- ultralytics (Python): 8.4.207 was released" in text, text
    assert "| rfdetr (Python) | 1.11.2 | 1.12.0 | 1.12.0 was released" in text, text
    assert "| scipy (Python) | 1.17.1 | 1.18.1 | 1.18.1 needs Python" in text, text
    assert "| ONNX Runtime (cpu/directml/coreml) | 1.24.4 | 1.31.1 | DirectML |" in text, text
    assert "onnx (Python)" not in text and "- uv: could not read upstream" in text, text
    text, any_due = upstream_report([row("rfdetr (Python)", "1.11.2", "1.11.2", "1.12.0", "too new")])
    assert not any_due and text.startswith("No pin is due for a bump.\n\nNewer, but not due yet:"), text
    text, any_due = upstream_report([row("onnx (Python)", "1.23.2", "1.23.2", "1.23.2")])
    assert not any_due and text == "Every pin is at its latest upstream release.\n", text

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
    ap.add_argument("command", nargs="?", choices=("consistency", "upstream", "audit", "conflicts"))
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
    if a.command == "upstream":
        text, code = upstream(ROOT, a.today or dt.date.today())
    elif a.command == "conflicts":
        text, code = conflicts(ROOT)
    else:
        text, code = audit(ROOT)
    sys.stdout.write(text)
    if a.out:
        a.out.write_text(text, encoding="utf-8")
    return code


if __name__ == "__main__":
    sys.exit(main())
