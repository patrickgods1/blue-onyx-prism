#!/usr/bin/env bash
# Build the unsigned macOS arm64 installer package.
# Usage: packaging/macos/build-pkg.sh <version> <bin-dir> <out.pkg>
#   <bin-dir> holds the release binaries blue-onyx-prism, blue-onyx-prism-benchmark and
#   blue-onyx-prism-tray.
# Needs macOS (pkgbuild, productbuild). Runtimes are not bundled: the service downloads them.
set -euo pipefail

if [ "$#" -ne 3 ]; then
  echo "Usage: $0 <version> <bin-dir> <out.pkg>" >&2
  exit 2
fi
VERSION="$1"
BIN_DIR="$(cd "$2" && pwd)"
OUT="$3"
case "$OUT" in /*) ;; *) OUT="$PWD/$OUT" ;; esac

HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$HERE/../.." && pwd)"
for f in blue-onyx-prism blue-onyx-prism-benchmark blue-onyx-prism-tray; do
  [ -f "$BIN_DIR/$f" ] || { echo "Missing $BIN_DIR/$f" >&2; exit 1; }
done
command -v pkgbuild >/dev/null && command -v productbuild >/dev/null && command -v osacompile >/dev/null || {
  echo "pkgbuild/productbuild/osacompile not found: run on macOS." >&2; exit 1; }

WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
ROOT="$WORK/root"
PROG="$ROOT/usr/local/blue-onyx-prism"
mkdir -p "$PROG/scripts" "$ROOT/Library/LaunchDaemons" "$ROOT/Library/LaunchAgents" "$ROOT/Applications" "$WORK/resources" "$WORK/pkgs"

install -m 0755 "$BIN_DIR/blue-onyx-prism" "$BIN_DIR/blue-onyx-prism-benchmark" "$PROG/"
install -m 0644 "$REPO/LICENSE" "$REPO/THIRD_PARTY_LICENSES.md" "$REPO/README.md" "$PROG/"
install -m 0755 "$REPO/scripts/uninstall_launchd.sh" "$PROG/scripts/"
install -m 0644 "$REPO/scripts/launchd_common.sh" "$PROG/scripts/"
# The plist already names the default paths (/usr/local/blue-onyx-prism, the data root, the log
# dir): install_launchd.sh only substitutes a custom install dir, which the pkg never has.
install -m 0644 "$REPO/deploy/com.blueonyx.prism.plist" "$ROOT/Library/LaunchDaemons/"
install -m 0644 "$REPO/deploy/com.blueonyx.prism.tray.plist" "$ROOT/Library/LaunchAgents/"

# "Blue Onyx Prism.app": the menu bar app (LSUIElement: no Dock icon), started at login by the
# LaunchAgent above. Ad-hoc signed like the uninstaller.
TRAY_APP="$ROOT/Applications/Blue Onyx Prism.app"
mkdir -p "$TRAY_APP/Contents/MacOS"
install -m 0755 "$BIN_DIR/blue-onyx-prism-tray" "$TRAY_APP/Contents/MacOS/"
cat > "$TRAY_APP/Contents/Info.plist" <<PLIST
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0">
<dict>
  <key>CFBundleName</key>
  <string>Blue Onyx Prism</string>
  <key>CFBundleDisplayName</key>
  <string>Blue Onyx Prism</string>
  <key>CFBundleIdentifier</key>
  <string>com.blueonyx.prism.tray</string>
  <key>CFBundleExecutable</key>
  <string>blue-onyx-prism-tray</string>
  <key>CFBundlePackageType</key>
  <string>APPL</string>
  <key>CFBundleShortVersionString</key>
  <string>$VERSION</string>
  <key>CFBundleVersion</key>
  <string>$VERSION</string>
  <key>LSMinimumSystemVersion</key>
  <string>12.0</string>
  <key>LSUIElement</key>
  <true/>
  <key>NSHighResolutionCapable</key>
  <true/>
</dict>
</plist>
PLIST
plutil -lint "$TRAY_APP/Contents/Info.plist" >/dev/null
codesign --force --sign - "$TRAY_APP"
codesign --verify --strict "$TRAY_APP"

# "Uninstall Blue Onyx Prism.app": runs scripts/uninstall_launchd.sh behind the admin password
# prompt (macOS has no uninstall for pkg installs). osacompile signs it ad hoc; the bundle id is
# set afterwards, so sign again.
APP="$ROOT/Applications/Uninstall Blue Onyx Prism.app"
osacompile -o "$APP" "$HERE/uninstaller.applescript"
/usr/libexec/PlistBuddy -c "Add :CFBundleIdentifier string com.blueonyx.prism.uninstaller" \
  -c "Add :CFBundleShortVersionString string $VERSION" "$APP/Contents/Info.plist"
codesign --force --sign - "$APP"
codesign --verify --strict "$APP"

# Bundles in a pkg are relocatable by default: the installer would update a copy of the app found
# elsewhere (e.g. one the user moved) instead of installing into /Applications.
COMPONENTS="$WORK/components.plist"
pkgbuild --analyze --root "$ROOT" "$COMPONENTS"
i=0
while /usr/libexec/PlistBuddy -c "Print :$i" "$COMPONENTS" >/dev/null 2>&1; do
  /usr/libexec/PlistBuddy -c "Set :$i:BundleIsRelocatable false" "$COMPONENTS"
  i=$((i + 1))
done

SCRIPTS="$WORK/scripts"
mkdir -p "$SCRIPTS"
install -m 0755 "$HERE/scripts/preinstall" "$HERE/scripts/postinstall" "$SCRIPTS/"

cp "$HERE/resources/"*.html "$WORK/resources/"
cp "$REPO/LICENSE" "$WORK/resources/LICENSE"

pkgbuild --root "$ROOT" --scripts "$SCRIPTS" --component-plist "$COMPONENTS" \
  --identifier com.blueonyx.prism --version "$VERSION" \
  --install-location / --ownership recommended \
  "$WORK/pkgs/component.pkg"

mkdir -p "$(dirname "$OUT")"
productbuild --distribution "$HERE/distribution.xml" \
  --resources "$WORK/resources" --package-path "$WORK/pkgs" \
  --version "$VERSION" "$OUT"
echo "Built $OUT"
