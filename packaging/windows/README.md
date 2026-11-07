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
`msiexec /i blue-onyx-prism.msi /qn /norestart PORT=32169` (without `/norestart`, a silent install that finds a file in use may restart the machine). The install folder can be changed in the UI or with `INSTALLFOLDER=`.

`REMOVEDATA=1` on uninstall deletes the data folder (below). Only the value `1` does. `KEEPDATA=1` asked for keeping
that folder before it became the default: it is still accepted, so old scripts work, and wins over `REMOVEDATA=1`.

## Uninstall

An uninstall removes the program files, the service, the firewall rule, the event source, the Run
value, the shortcuts, `HKLM\SOFTWARE\BlueOnyxPrism` and, when the program folder is the default
`C:\Program Files\Blue Onyx Prism`, anything the program created there (a folder the user picked
keeps files the MSI did not install).

It keeps the data folder `C:\ProgramData\BlueOnyxPrism`: config, downloaded runtimes, models, logs
and, unless they were pointed elsewhere, the dataset, reviewed labels and training runs, which no
download brings back. A reinstall picks the folder up and sets its DACL again. To delete it too,
which cannot be undone:

```powershell
msiexec /x <file>.msi REMOVEDATA=1    # the MSI of the installed version; then nothing is left behind
Remove-Item -LiteralPath C:\ProgramData\BlueOnyxPrism -Recurse -Force    # or by hand, after an uninstall (elevated)
```

Removing the old version during an upgrade keeps both folders' data. The tray's lock file in the
user's temp folder is deleted when the tray exits (also when the uninstall terminates it).

An installed version uninstalls by its own rules, from the MSI Windows cached for it. Up to 0.2.0
that means the data folder goes unless `KEEPDATA=1` is passed: upgrade first, then uninstall the new
version. `smoke-test.ps1` checks all of this for the current MSI: a plain uninstall, `KEEPDATA=1`
and `REMOVEDATA=1`.

## Coexistence with scripts\install_service.ps1

The MSI refuses to install when the service exists and was not installed by this MSI (marker value
`HKLM\SOFTWARE\BlueOnyxPrism\Installer`). Run `scripts\uninstall_service.ps1` first. Upgrades of the MSI itself are not blocked.
The MSI does not change `ServicesPipeTimeout`.

## CI smoke test

`smoke-test.ps1 -Msi <path> [-Port 32168]` (elevated): installs, checks service/account/HTTP/config, the data folder's
DACL, the required privileges, the firewall rule's scope and profiles and the tray exe, Run value and shortcuts (the tray
is not run), uninstalls, checks cleanup. Four rounds: `REMOVEDATA=1` leaves nothing behind; over the deleted folder, a plain
uninstall keeps only the data folder, a seeded dataset file included; over that data with an older version's ACL and a
token, both folders' DACLs are corrected, the data is intact and `KEEPDATA=1` keeps it; a last install and `REMOVEDATA=1`
remove everything, the protected token folder too.
