# macOS installer package

Unsigned arm64 `.pkg` that installs the same layout as `scripts/install_launchd.sh`:

- program: `/usr/local/blue-onyx-prism` (binaries, licenses, `scripts/uninstall_launchd.sh`, `scripts/launchd_common.sh`)
- data: `/usr/local/var/blue-onyx-prism` (config, models, cache, downloads), owned by the role account `_blueonyx`
- logs: `/var/log/blue-onyx-prism` (rotated through `/etc/newsyslog.d/blue-onyx-prism.conf`)
- daemon: `/Library/LaunchDaemons/com.blueonyx.prism.plist`

Runtimes are not bundled. The service downloads OpenVINO / ONNX Runtime and the default model on
first start into `download_dir`. The seeded config sets `download_dir`, `cache_dir` and `log_path`
only; an existing config is never overwritten, so an upgrade keeps settings and models.

## Build (on macOS)

```sh
packaging/macos/build-pkg.sh <version> <bin-dir> blue-onyx-prism-<version>-macos-aarch64.pkg
```

`<bin-dir>` holds the release `blue-onyx-prism` and `blue-onyx-prism-benchmark`. Uses `pkgbuild`
and `productbuild`; the postinstall helpers come from `scripts/launchd_common.sh`.

Test on a macOS machine with sudo: `packaging/macos/smoke-test.sh <pkg>` (installs, waits for
`/v1/status`, uninstalls with `--purge`).

## Install an unsigned package

Gatekeeper blocks a package that is not signed or notarized. Double-click it, dismiss the warning,
then open System Settings > Privacy & Security, scroll to the message about the blocked package and
click "Open Anyway". Or from Terminal: `sudo installer -pkg <file>.pkg -target /`.

After install open http://127.0.0.1:32168/ (status and configuration).

## Uninstall

```sh
sudo /usr/local/blue-onyx-prism/scripts/uninstall_launchd.sh            # keep data, logs, account
sudo /usr/local/blue-onyx-prism/scripts/uninstall_launchd.sh --purge    # also delete them
sudo rm -rf /usr/local/blue-onyx-prism                                  # the program files
```

The script also forgets the installer receipt (`pkgutil --forget com.blueonyx.prism`).
