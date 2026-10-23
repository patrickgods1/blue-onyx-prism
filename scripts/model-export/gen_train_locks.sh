#!/bin/sh
# Regenerate the hashed package locks of the on-demand training environment from
# scripts/requirements-model-train.txt. The service embeds the locks (src/resources/export/train.rs)
# and installs them with `uv pip install --require-hashes --no-deps --only-binary :all:` (plus
# `--torch-backend cu126|cu130` for the CUDA ones), so every wheel is checked against these hashes.
#
#   scripts/model-export/gen_train_locks.sh <path to uv> [exclude-newer timestamp]
#
# Use the pinned uv (catalog `tool:uv`, src/resources/catalog/). The timestamp freezes the
# transitive dependencies; keep it at least two weeks in the past and update the pins in
# requirements-model-train.txt in the same commit.
#
# All five are resolved on any machine (uv cross-platform resolution, `--python-platform`); the
# CUDA ones were generated on macOS arm64 this way.
#
# - locks/train-macos-arm64.txt: macOS 14+ arm64, PyTorch from PyPI (MPS).
# - locks/train-cu126-{windows,linux}-x86_64.txt: PyTorch CUDA 12.6 builds (compute capability
#   5.0 to 9.0 kernels; the service offers 6.1+), any driver with CUDA 12.
# - locks/train-cu130-{windows,linux}-x86_64.txt: PyTorch CUDA 13.0 builds (7.5 to 12.0), for
#   Blackwell (10.0, 12.0), which the 12.6 builds have no kernels for; needs a CUDA 13 driver.
set -eu
UV="$1"
EXCLUDE_NEWER="${2:-2026-10-08T00:00:00Z}"
DIR="$(cd "$(dirname "$0")" && pwd)"
IN="$DIR/../requirements-model-train.txt"
OUT="$DIR/locks"
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
mkdir -p "$OUT"

compile() { # <python-platform> <output> [extra args]
  t="$1"; o="$2"; shift 2
  MACOSX_DEPLOYMENT_TARGET=14.0 UV_NO_CONFIG=1 UV_CACHE_DIR="$TMP/cache" "$UV" pip compile "$IN" \
    --python-version 3.11 --python-platform "$t" --generate-hashes --only-binary :all: \
    --exclude-newer "$EXCLUDE_NEWER" --no-header --no-annotate -q -o "$o" "$@"
}

compile aarch64-apple-darwin "$OUT/train-macos-arm64.txt"
for cu in cu126 cu130; do
  compile x86_64-pc-windows-msvc "$OUT/train-$cu-windows-x86_64.txt" --torch-backend "$cu"
  compile x86_64-unknown-linux-gnu "$OUT/train-$cu-linux-x86_64.txt" --torch-backend "$cu"
done
for f in "$OUT"/train-*.txt; do echo "$f: $(grep -c '==' "$f") packages"; done
