# Windows installer (MSI)

Per-machine MSI built with WiX Toolset v7 (pinned 7.0.0; v4+ schema in `main.wxs`).

## Build locally

```powershell
cargo build --release --locked --bins
.\packaging\windows\build-msi.ps1 -Version 0.1.0 -BinDir target\release -OutFile dist\blue-onyx-prism-0.1.0.msi
```

`build-msi.ps1` installs the `wix` dotnet tool and the Util/Firewall/UI extensions (all 7.0.0) when
`wix` is not on PATH, generates `obj\License.rtf` from `LICENSE`, and runs `wix build -acceptEula wix7`
(WiX 7 requires accepting its OSMF EULA). Needs a .NET SDK (8+). `-Version` must be numeric `x.y.z`.

## What it installs

- `C:\Program Files\Blue Onyx Prism`: service exe, CLI, tray exe (`blue-onyx-prism-tray.exe`), benchmark exe, LICENSE, THIRD_PARTY_LICENSES.md, README.md.
- Service `BlueOnyxPrismService` (auto start, `NT AUTHORITY\LocalService`), started `--config "C:\ProgramData\BlueOnyxPrism\blue_onyx_prism_config_service.json"`;
  restarts after 5 s / 30 s / 120 s (count reset after 1 h).
- `C:\ProgramData\BlueOnyxPrism`: config, models, cache, logs; Administrators/SYSTEM full control, LocalService modify.
- Application event log source `BlueOnyxPrism`, firewall rule `Blue Onyx Prism (<PORT>)` (TCP, any scope, all profiles),
  Start Menu shortcuts "Blue Onyx Prism" (the tray icon app) and "Blue Onyx Prism Web UI" (`http://127.0.0.1:32168/`).
- Tray app: `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run\BlueOnyxPrismTray` starts it at login for every user.
  A running tray is terminated (WiX `CloseApplication`) before an upgrade or uninstall replaces/removes its exe. The
  installer does not launch it at the end (it starts at the next login or from the Start Menu).

## Properties

`PORT` (default 32168): firewall rule port only; also set `port` in the config if you change it. Example:
`msiexec /i blue-onyx-prism.msi /qn PORT=32169`. The install folder can be changed in the UI or with `INSTALLFOLDER=`.

## Uninstall

A full uninstall leaves nothing behind: program files, the service, the firewall rule, the event
source, the Run value, the shortcuts, `HKLM\SOFTWARE\BlueOnyxPrism`, the data folder
`C:\ProgramData\BlueOnyxPrism` (config, downloaded runtimes, models, logs) and, when the program
folder is the default `C:\Program Files\Blue Onyx Prism`, anything the program created there
(a folder the user picked keeps files the MSI did not install). `msiexec /x <file>.msi KEEPDATA=1`
keeps the data folder. Removing the old version during an upgrade keeps both folders' data. The
tray's lock file in the user's temp folder is deleted when the tray exits (also when the
uninstall terminates it). `smoke-test.ps1` checks all of this, with and without `KEEPDATA=1`.

## Coexistence with scripts\install_service.ps1

The MSI refuses to install when the service exists and was not installed by this MSI (marker value
`HKLM\SOFTWARE\BlueOnyxPrism\Installer`). Run `scripts\uninstall_service.ps1` first. Upgrades of the MSI itself are not blocked.
The MSI does not change `ServicesPipeTimeout`.

## CI smoke test

`smoke-test.ps1 -Msi <path> [-Port 32168]` (elevated): installs, checks service/account/HTTP/config and the tray exe, Run value and shortcuts (the tray is not run), uninstalls, checks cleanup.
