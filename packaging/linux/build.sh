#!/usr/bin/env bash
# Build the .deb and .rpm from binaries that `cargo build --release --locked --bins --target T`
# already produced (they are stripped by the release profile).
# Usage: packaging/linux/build.sh <target-triple> <out-dir>
# Output: <out-dir>/blue-onyx-prism_<ver>-1_<deb arch>.deb and blue-onyx-prism-<ver>-1.<rpm arch>.rpm
set -euo pipefail

CARGO_DEB_VERSION=3.8.0
CARGO_GENERATE_RPM_VERSION=0.21.0

if [ $# -ne 2 ]; then
  echo "Usage: $0 <target-triple> <out-dir>" >&2
  exit 2
fi
TARGET="$1"
OUT="$2"
case "$TARGET" in
  x86_64-*) DEB_ARCH=amd64; RPM_ARCH=x86_64 ;;
  aarch64-*) DEB_ARCH=arm64; RPM_ARCH=aarch64 ;;
  *) echo "Unsupported target $TARGET" >&2; exit 2 ;;
esac

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
VERSION="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
mkdir -p "$OUT"
OUT="$(cd "$OUT" && pwd)"

for b in blue-onyx-prism blue-onyx-prism-benchmark; do
  [ -x "target/$TARGET/release/$b" ] || { echo "Missing target/$TARGET/release/$b: cargo build --release --locked --bins --target $TARGET first." >&2; exit 1; }
done

# The unit comes from deploy/ with the same substitution install_systemd.sh applies.
# shellcheck source=../../scripts/systemd_common.sh
. "$ROOT/scripts/systemd_common.sh"
mkdir -p packaging/linux/build
bop_install_unit deploy/blue-onyx-prism.service packaging/linux/build/blue-onyx-prism.service /opt/blue-onyx-prism

have_tool() { cargo "$1" --version >/dev/null 2>&1; }
have_tool deb || cargo install cargo-deb --version "$CARGO_DEB_VERSION" --locked
have_tool generate-rpm || cargo install cargo-generate-rpm --version "$CARGO_GENERATE_RPM_VERSION" --locked

DEB="$OUT/blue-onyx-prism_${VERSION}-1_${DEB_ARCH}.deb"
RPM="$OUT/blue-onyx-prism-${VERSION}-1.${RPM_ARCH}.rpm"

# --no-strip: the release profile already strips; --no-build: use the binaries as built.
cargo deb --no-build --no-strip --target "$TARGET" --output "$DEB"
cargo generate-rpm --target "$TARGET" --output "$RPM"

ls -l "$DEB" "$RPM"
