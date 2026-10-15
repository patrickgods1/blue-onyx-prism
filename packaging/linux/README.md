# Linux packages (.deb / .rpm)

Installs the service like `scripts/install_systemd.sh`: binaries `blue-onyx-prism` and
`blue-onyx-prism-benchmark` (plus LICENSE, THIRD_PARTY_LICENSES.md, README.md and `scripts/`) in
`/opt/blue-onyx-prism/` (root-owned), data and config in `/var/lib/blue-onyx-prism/`, system user
`blueonyx` (groups `render`, `video`), unit `/usr/lib/systemd/system/blue-onyx-prism.service`
enabled and started. OpenVINO / ONNX Runtime are not bundled: the service downloads them on first
start into `download_dir` (the seeded config sets `download_dir` and `cache_dir` only).

Build (after `cargo build --release --locked --bins --target T`):

    packaging/linux/build.sh x86_64-unknown-linux-gnu dist/

Smoke test (Ubuntu with sudo + systemd): `packaging/linux/smoke-test.sh dist/*.deb dist/*.rpm`.

Files:

- `build.sh` generates the unit from `deploy/blue-onyx-prism.service` (into the git-ignored
  `build/`, same substitution as `install_systemd.sh`), installs the pinned `cargo-deb` /
  `cargo-generate-rpm` if missing and runs them. Metadata is in `Cargo.toml`
  (`[package.metadata.deb]`, `[package.metadata.generate-rpm]`).
- `blue-onyx-prism.sysusers` -> `/usr/lib/sysusers.d/`; the scriptlet runs `systemd-sysusers`, with
  `useradd` as fallback.
- `pkg-scriptlet.sh` -> `/opt/blue-onyx-prism/scripts/pkg_scriptlet.sh`: the one implementation of
  post-install / post-upgrade / pre-remove, built on `scripts/systemd_common.sh` (also used by
  `install_systemd.sh`). `debian/{postinst,prerm,postrm}` and `rpm/*.sh` are thin wrappers.
- No `systemd-units` in cargo-deb: the scriptlet does enable/start itself, so nothing is handled
  twice (`#DEBHELPER#` stays in the scripts but expands to nothing).

Behavior: config is never overwritten; install enables and starts, upgrade only `try-restart`s;
removal stops and disables (not on upgrade). `apt remove` also deletes what the program created in
`/opt/blue-onyx-prism` and keeps `/var/lib/blue-onyx-prism` and the user for a reinstall; `apt purge`
deletes those too (like `uninstall_systemd.sh --purge`). RPM has no purge, so an erase (`dnf remove`,
not an upgrade) leaves nothing behind: `/opt/blue-onyx-prism`, `/var/lib/blue-onyx-prism` and the user
and group. `smoke-test.sh` checks both, the RPM in a Fedora container, and searches the filesystem for
leftovers (names and files owned by the deleted user).
Scriptlets skip `systemctl` when systemd is not running (containers, chroots).
