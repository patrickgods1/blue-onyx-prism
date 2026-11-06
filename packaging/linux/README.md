# Linux packages (.deb / .rpm)

Installs the service like `scripts/install_systemd.sh`: binaries `blue-onyx-prism` and
`blue-onyx-prism-benchmark` (plus LICENSE, THIRD_PARTY_LICENSES.md, README.md and `scripts/`, which
also holds the by-hand training scripts) in
`/opt/blue-onyx-prism/` (root-owned), data and config in `/var/lib/blue-onyx-prism/`, system user
`blueonyx` (groups `render`, `video`), unit `/usr/lib/systemd/system/blue-onyx-prism.service`
enabled and started. OpenVINO / ONNX Runtime are not bundled: the service downloads them on first
start into `download_dir` (the seeded config sets `download_dir` and `cache_dir` only).

Build (after `cargo build --release --locked --bins --target T`):

    packaging/linux/build.sh x86_64-unknown-linux-gnu dist/

Smoke test (Ubuntu with sudo + systemd): `packaging/linux/smoke-test.sh dist/*.deb dist/*.rpm`.
`packaging/linux/test-scriptlets.sh` runs the maintainer scripts' decisions against a temp folder with
stubbed system commands: no root, no package, any OS with bash.

Files:

- `build.sh` generates the unit from `deploy/blue-onyx-prism.service` (into the git-ignored
  `build/`, same substitution as `install_systemd.sh`), installs the pinned `cargo-deb` /
  `cargo-generate-rpm` if missing and runs them. Metadata is in `Cargo.toml`
  (`[package.metadata.deb]`, `[package.metadata.generate-rpm]`).
- `blue-onyx-prism.sysusers` -> `/usr/lib/sysusers.d/`; the scriptlet runs `systemd-sysusers`, with
  `useradd` as fallback.
- `pkg-scriptlet.sh` -> `/opt/blue-onyx-prism/scripts/pkg_scriptlet.sh`: the one implementation of
  post-install / post-upgrade / pre-remove, built on `scripts/systemd_common.sh` (also used by
  `install_systemd.sh`). `debian/{preinst,postinst,prerm,postrm}` and `rpm/*.sh` are thin wrappers.
- No `systemd-units` in cargo-deb: the scriptlet does enable/start itself, so nothing is handled
  twice (`#DEBHELPER#` stays in the scripts but expands to nothing).

Behavior: config is never overwritten; install enables and starts, upgrade only `try-restart`s, so a
unit the admin disabled stays disabled; removal stops and disables (not on upgrade).

`apt remove` and `dnf remove` delete `/opt/blue-onyx-prism`, with what the program created there, and
keep `/var/lib/blue-onyx-prism` (config, models, dataset, reviewed labels, training runs) and the
user for a reinstall, and both say so. `apt purge` deletes those too, like
`uninstall_systemd.sh --purge`. RPM has no purge, so the erase prints the commands:
`sudo rm -rf /var/lib/blue-onyx-prism && sudo userdel blueonyx`. An RPM of 0.2.0 or older deleted the
data on an erase, and an installed package erases by its own scripts: upgrade first.

A reinstall over kept data works like a first install, and `apt` needs help for it: `apt remove` leaves
the package in dpkg's config-files state, so the postinst gets the old version as on an upgrade,
though the unit was disabled. `debian/preinst` sees `install <old-version>` and leaves
`/opt/blue-onyx-prism/.reinstall` for the postinst. A data root whose owner is not the current
`blueonyx` (the user was deleted by hand and made again, or the folder came from a backup) is chowned
as a whole first, before `install -d` corrects only its top folders and hides the mismatch from
`StateDirectory=` as well.

`smoke-test.sh` checks all of this, the RPM in a Fedora container, including a reinstall with another
uid, and searches the filesystem for leftovers (names and files owned by the deleted user).
Scriptlets skip `systemctl` when systemd is not running (containers, chroots).
