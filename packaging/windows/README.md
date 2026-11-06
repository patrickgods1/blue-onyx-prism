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
  restarts after 5 s / 30 s / 120 s (count reset after 1 h). Its only required privilege is `SeChangeNotifyPrivilege`:
  the SCM removes every other one from the process, as `scripts\install_service.ps1` does.
- `C:\ProgramData\BlueOnyxPrism`: config, models, cache, logs. Its DACL is protected, so nothing is inherited from
  `ProgramData` (whose ACL lets `Users` create files): Administrators/SYSTEM full control, LocalService modify, Users
  read (the tray reads the port from the config). Commands that write here, such as `hf-login` or a `fetch` by hand,
  need an elevated terminal. Every install, repair and upgrade sets this again, so an upgrade corrects an older install.
  `tools\hf` has its own protected DACL, because it holds the Hugging Face token of `hf-login`, and Users cannot read it.
- Application event log source `BlueOnyxPrism`, firewall rule `Blue Onyx Prism (<PORT>)` (see below),
  Start Menu shortcuts "Blue Onyx Prism" (the tray icon app) and "Blue Onyx Prism Web UI" (`http://127.0.0.1:32168/`).
- Tray app: `HKLM\SOFTWARE\Microsoft\Windows\CurrentVersion\Run\BlueOnyxPrismTray` starts it at login for every user.
  A running tray is terminated (WiX `CloseApplication`) before an upgrade or uninstall replaces/removes its exe. The
  installer does not launch it at the end (it starts at the next login or from the Start Menu).

## Firewall

The rule `Blue Onyx Prism (<PORT>)` allows TCP `<PORT>` from the local subnet only, on the Domain and Private network
profiles. A Blue Iris on the same subnet, or on the same PC, needs nothing more. Two cases do:

- **Blue Iris on another subnet or VLAN.** Add your own rule, for that machine's address, in an elevated PowerShell:

  ```powershell
  New-NetFirewallRule -DisplayName "Blue Onyx Prism (Blue Iris)" -Direction Inbound -Action Allow `
      -Protocol TCP -LocalPort 32168 -RemoteAddress 192.168.20.5 -Profile Domain,Private
  ```

  An uninstall does not remove this rule: `Remove-NetFirewallRule -DisplayName "Blue Onyx Prism (Blue Iris)"`.
- **This PC's network is set to Public.** Windows often picks Public for a new network, and the rule does not apply
  there. Check with `Get-NetConnectionProfile`, and make a trusted network Private:
  `Set-NetConnectionProfile -InterfaceAlias Ethernet -NetworkCategory Private`.

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

`smoke-test.ps1 -Msi <path> [-Port 32168]` (elevated): installs, checks service/account/HTTP/config, the data folder's
DACL, the required privileges, the firewall rule's scope and profiles and the tray exe, Run value and shortcuts (the tray
is not run), uninstalls, checks cleanup. A third round installs over data with an older version's ACL and a token, and
checks that both folders' DACLs are corrected and that the full uninstall still removes them.
