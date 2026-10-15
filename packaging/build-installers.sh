#!/usr/bin/env bash
# Build this runner's native installer(s) from binaries already built with
# `cargo build --release --locked --bins --target <target>`. Used by release.yml and packaging.yml.
# Usage: packaging/build-installers.sh <target-triple> <out-dir>
# Output (<ver> from Cargo.toml):
#   Windows: blue-onyx-prism-<ver>-windows-x86_64.msi
#   Linux:   blue-onyx-prism_<ver>-1_amd64.deb, blue-onyx-prism-<ver>-1.x86_64.rpm
#   macOS:   blue-onyx-prism-<ver>-macos-aarch64.pkg
# Each installer gets a <file>.sha256 next to it.
set -euo pipefail

if [ $# -ne 2 ]; then
  echo "Usage: $0 <target-triple> <out-dir>" >&2
  exit 2
fi
TARGET="$1"
OUT="$2"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
VERSION="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
BIN="target/$TARGET/release"
mkdir -p "$OUT"

case "$TARGET" in
  *-windows-*)
    pwsh -NoProfile -File packaging/windows/build-msi.ps1 -Version "$VERSION" -BinDir "$BIN" \
      -OutFile "$OUT/blue-onyx-prism-$VERSION-windows-x86_64.msi"
    ;;
  *-linux-*)
    packaging/linux/build.sh "$TARGET" "$OUT"
    ;;
  aarch64-apple-darwin)
    packaging/macos/build-pkg.sh "$VERSION" "$BIN" "$OUT/blue-onyx-prism-$VERSION-macos-aarch64.pkg"
    ;;
  *)
    echo "No installer for target $TARGET" >&2
    exit 2
    ;;
esac

# Code signing goes here once there is a certificate: the MSI via SignPath or Azure Artifact
# Signing (signtool), the pkg via productsign + xcrun notarytool + stapler. Sign before hashing.

cd "$OUT"
for f in *.msi *.deb *.rpm *.pkg; do
  [ -f "$f" ] || continue
  if command -v sha256sum >/dev/null; then sha256sum "$f" > "$f.sha256"; else shasum -a 256 "$f" > "$f.sha256"; fi
  cat "$f.sha256"
done
