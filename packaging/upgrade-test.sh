#!/usr/bin/env bash
# Upgrade from the latest published release to this build's installer (CI only: it changes the
# machine; run after smoke-test-installers.sh, which leaves it clean). Downloads the published
# installer for this runner with gh (checked against its .sha256 asset), then runs
# packaging/windows/upgrade-test.ps1 (MSI) or packaging/upgrade-test-unix.sh (.deb, .pkg).
# Usage: packaging/upgrade-test.sh <target-triple> <dir with this build's installers> [tag]
# tag: a release tag, default the latest published release (drafts and pre-releases skipped).
# Env: GH_TOKEN, GITHUB_REPOSITORY (else the repository of the checkout); BOP_UPGRADE_DOWNLOAD_ONLY=1
# stops after downloading and checking the published installer (installs nothing).
set -euo pipefail

if [ $# -lt 2 ] || [ $# -gt 3 ]; then
  echo "Usage: $0 <target-triple> <dir with this build's installers> [tag]" >&2
  exit 2
fi
TARGET="$1"
DIST="$(cd "$2" && pwd)"
TAG="${3:-latest}"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
REPO="${GITHUB_REPOSITORY:-$(gh repo view --json nameWithOwner --jq .nameWithOwner)}"
VERSION="$(grep -m1 '^version' Cargo.toml | sed -E 's/.*"([^"]+)".*/\1/')"
BIN="$ROOT/target/$TARGET/release"

one() { # exactly one file matching the glob
  # shellcheck disable=SC2086 # the glob must expand
  set -- $1
  [ $# -eq 1 ] && [ -f "$1" ] || { echo "expected one installer, found: $*" >&2; exit 1; }
  echo "$1"
}

if [ "$TAG" = latest ]; then
  TAG="$(gh release list --repo "$REPO" --exclude-drafts --exclude-pre-releases --limit 1 --json tagName --jq '.[0].tagName // ""')" \
    || { echo "::error::Could not list the releases of $REPO (see above)." >&2; exit 1; }
  if [ -z "$TAG" ]; then
    echo "::error::$REPO has no published release to upgrade from. Publish one first, or pass a tag." >&2
    exit 1
  fi
fi
gh release view "$TAG" --repo "$REPO" --json tagName >/dev/null \
  || { echo "::error::$REPO has no release $TAG to upgrade from (see above)." >&2; exit 1; }
echo "Upgrading from release $TAG to this build ($VERSION)"

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*) pattern='*-windows-x86_64.msi' ;;
  Linux) pattern='*_amd64.deb' ;;
  Darwin) pattern='*-macos-aarch64.pkg' ;;
  *) echo "unsupported OS $(uname -s)" >&2; exit 2 ;;
esac
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
OLD_DIR="$WORK/release-$TAG"
mkdir -p "$OLD_DIR"
"$ROOT/scripts/ci/retry.sh" 4 20 gh release download "$TAG" --repo "$REPO" --dir "$OLD_DIR" \
  --pattern "$pattern" --pattern "$pattern.sha256" \
  || { echo "::error::Release $TAG of $REPO has no asset matching $pattern (or the download failed, see above)." >&2; exit 1; }
OLD="$(one "$OLD_DIR/$pattern")"
[ -f "$OLD.sha256" ] || { echo "::error::Release $TAG has $(basename "$OLD") but no $(basename "$OLD").sha256." >&2; exit 1; }
(cd "$OLD_DIR" && if command -v sha256sum >/dev/null; then sha256sum -c "$(basename "$OLD").sha256"; else shasum -a 256 -c "$(basename "$OLD").sha256"; fi) \
  || { echo "::error::$(basename "$OLD") does not match its published .sha256." >&2; exit 1; }

echo "$(basename "$OLD") matches its published .sha256"
if [ "${BOP_UPGRADE_DOWNLOAD_ONLY:-}" = 1 ]; then
  echo "BOP_UPGRADE_DOWNLOAD_ONLY=1: stopping after the download (nothing installed)."
  exit 0
fi

# Paths for pwsh: Git Bash's /d/a/... form as D:\a\...
winpath() { if command -v cygpath >/dev/null; then cygpath -w "$1"; else echo "$1"; fi; }

# newer A B: true when version A > version B (numeric x.y.z, dot by dot).
newer() {
  awk -v a="$1" -v b="$2" 'BEGIN {
    na = split(a, x, "."); nb = split(b, y, "."); n = na > nb ? na : nb
    for (i = 1; i <= n; i++) { if (x[i] + 0 > y[i] + 0) exit 0; if (x[i] + 0 < y[i] + 0) exit 1 }
    exit 1
  }'
}

case "$(uname -s)" in
  MINGW* | MSYS* | CYGWIN*)
    NEW="$(one "$DIST/*.msi")"
    OLD_VERSION="${TAG#v}"
    if ! newer "$VERSION" "$OLD_VERSION"; then
      # Main between releases has the released version, and an MSI of the same version is no
      # upgrade (the old product would stay beside it). The same binaries as a patch release
      # above it stand in; MSI compares only the first three fields.
      IFS=. read -r major minor patch <<< "${OLD_VERSION%%-*}"
      STAND_IN="$major.$minor.$((patch + 1))"
      echo "This build ($VERSION) is not newer than $OLD_VERSION: building the same binaries as MSI $STAND_IN for the upgrade."
      NEW="$DIST/upgrade/blue-onyx-prism-$STAND_IN-windows-x86_64.msi"
      pwsh -NoProfile -File packaging/windows/build-msi.ps1 -Version "$STAND_IN" -BinDir "$(winpath "$BIN")" \
        -OutFile "$(winpath "$NEW")"
    fi
    pwsh -NoProfile -File packaging/windows/upgrade-test.ps1 -OldMsi "$(winpath "$OLD")" -NewMsi "$(winpath "$NEW")" \
      -NewServiceExe "$(winpath "$BIN/blue-onyx-prism-service.exe")" -LogDir "$(winpath "$DIST")"
    ;;
  Linux)
    packaging/upgrade-test-unix.sh "$OLD" "$(one "$DIST/*.deb")" "$BIN/blue-onyx-prism"
    ;;
  Darwin)
    packaging/upgrade-test-unix.sh "$OLD" "$(one "$DIST/*.pkg")" "$BIN/blue-onyx-prism"
    ;;
esac
