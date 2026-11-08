#!/usr/bin/env python3
"""Generate the Rust crate table of THIRD_PARTY_LICENSES.md from Cargo.lock.

Usage:
  python3 -I scripts/gen_third_party.py          # rewrite the table between the markers
  python3 -I scripts/gen_third_party.py --check  # exit 1 when the file differs (CI)

The table lists the dependency closure of the root package over normal and build
dependencies (dev-dependencies left out), for every target platform (default features, the only
optional one is `onnxruntime`), from `cargo metadata --format-version 1 --locked`.
"""
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FILE = ROOT / "THIRD_PARTY_LICENSES.md"
BEGIN = "<!-- BEGIN GENERATED CRATE TABLE (scripts/gen_third_party.py) -->"
END = "<!-- END GENERATED CRATE TABLE -->"


def table():
    out = subprocess.run(
        ["cargo", "metadata", "--format-version", "1", "--locked"],
        cwd=ROOT, check=True, capture_output=True, text=True,
    ).stdout
    meta = json.loads(out)
    pkgs = {p["id"]: p for p in meta["packages"]}
    nodes = {n["id"]: n for n in meta["resolve"]["nodes"]}
    root = meta["resolve"]["root"]
    seen, stack = set(), [root]
    while stack:
        i = stack.pop()
        if i in seen:
            continue
        seen.add(i)
        for d in nodes[i]["deps"]:
            if any(k["kind"] != "dev" for k in d["dep_kinds"]):
                stack.append(d["pkg"])
    seen.discard(root)
    rows = []
    for i in seen:
        p = pkgs[i]
        lic = p.get("license") or "UNKNOWN"
        src = p.get("repository") or p.get("homepage") or f"https://crates.io/crates/{p['name']}"
        rows.append((p["name"], p["version"], lic, src))
    rows.sort(key=lambda r: (r[0], tuple(int(x) if x.isdigit() else 0 for x in r[1].replace("-", ".").replace("+", ".").split("."))))
    lines = ["| Crate | Version | License | Source |", "|---|---|---|---|"]
    lines += [f"| {n} | {v} | {l} | {s} |" for n, v, l, s in rows]
    return "\n".join(lines)


def render(text):
    t = table()
    if BEGIN in text:
        head, rest = text.split(BEGIN, 1)
        _, tail = rest.split(END, 1)
    else:
        # first run: replace the existing table
        lines = text.split("\n")
        start = next(i for i, l in enumerate(lines) if l.startswith("| Crate |"))
        end = start
        while end < len(lines) and lines[end].startswith("|"):
            end += 1
        head = "\n".join(lines[:start]) + "\n"
        tail = "\n" + "\n".join(lines[end:])
    return f"{head}{BEGIN}\n{t}\n{END}{tail}"


def main():
    text = FILE.read_text()
    new = render(text)
    if "--check" in sys.argv:
        if new != text:
            print("THIRD_PARTY_LICENSES.md is out of date with Cargo.lock. Regenerate with: "
                  "python3 -I scripts/gen_third_party.py", file=sys.stderr)
            sys.exit(1)
        print("THIRD_PARTY_LICENSES.md is up to date")
    else:
        FILE.write_text(new)


main()
