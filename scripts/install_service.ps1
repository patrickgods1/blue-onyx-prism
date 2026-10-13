#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Installs Blue Onyx Prism as a Windows service (BlueOnyxPrismService).
.DESCRIPTION
    - Creates the Application event log source "BlueOnyxPrism"
    - Raises ServicesPipeTimeout to 600000 ms so the service has time to compile models on first start
      (machine-wide; the previous value is recorded under HKLM:\SOFTWARE\BlueOnyxPrism and
      uninstall_service.ps1 restores it)
    - Registers the service (auto start) pointing at blue-onyx-prism-service.exe, running as the
      virtual account "NT SERVICE\BlueOnyxPrismService" by default (no password, no rights beyond
      what this script grants)
    - Resets the install dir ACL: Administrators and SYSTEM full control, Users read and execute.
      The service account gets Modify only on its data: models, cache, tools, the runtime
      download dirs (openvino, onnxruntime, .downloads), the service config file and the files it
      keeps next to it, plus any download_dir / models_dir / cache_dir / save_image_path /
      log_path the config names. It can create new files in the install dir (staging dirs, the
      config's temp file) but cannot change the exe or the files shipped with it.
    - Restricts the service token: service SID type "unrestricted" (the NT SERVICE\BlueOnyxPrismService
      SID is in the token, so the ACL grants above apply under any account) and required
      privileges SeChangeNotifyPrivilege only (every other privilege is removed from the process)
    - Configures automatic restart on failure and a firewall rule for the listening port
.PARAMETER InstallDir
    Directory containing blue-onyx-prism-service.exe (default: the parent of this script's folder).
.PARAMETER Port
    TCP port to open in Windows Firewall (default 32168).
.PARAMETER Account
    Account to run the service as. Default: the virtual account NT SERVICE\BlueOnyxPrismService.
    "LocalSystem" restores the old behavior (full control of the machine; only if GPU access fails
    under the virtual account). "DOMAIN\user" or ".\user": you are prompted for the password; the
    account needs the "Log on as a service" right.
.PARAMETER NoStart
    Do not start the service after installing.
#>
param(
    [string]$InstallDir = (Split-Path -Parent $PSScriptRoot),
    [int]$Port = 32168,
    [string]$Account = "",
    [switch]$NoStart
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$ServiceName = "BlueOnyxPrismService"
$DisplayName = "Blue Onyx Prism"
$EventSource = "BlueOnyxPrism"
$VirtualAccount = "NT SERVICE\$ServiceName"
$ConfigName = "blue_onyx_prism_config_service.json"

# Run sc.exe and fail on a non-zero exit code (sc.exe never throws on its own; it reports
# errors on stdout, so stderr is not redirected: under "Stop" Windows PowerShell would throw on
# the first stderr line before the exit code could be read).
function Invoke-Sc {
    $out = & sc.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "sc.exe $($args -join ' ') failed (exit $LASTEXITCODE): $out"
    }
}

# Run icacls and fail on a non-zero exit code.
function Invoke-Icacls {
    $out = & icacls.exe @args
    if ($LASTEXITCODE -ne 0) {
        throw "icacls $($args -join ' ') failed (exit $LASTEXITCODE): $out"
    }
}

# Stop and delete a service if present; wait until the SCM has really removed it. While
# services.msc, Event Viewer or another tool holds a handle, `sc.exe delete` only marks the
# service for deletion and a new `sc.exe create` would fail with 1072, so this throws instead.
function Remove-ServiceIfPresent([string]$Name) {
    $svc = Get-Service -Name $Name -ErrorAction SilentlyContinue
    if (-not $svc) { return }
    if ($svc.Status -ne "Stopped") { Stop-Service $Name -Force; Start-Sleep -Seconds 2 }
    Invoke-Sc delete $Name
    for ($i = 0; $i -lt 60 -and (Get-Service -Name $Name -ErrorAction SilentlyContinue); $i++) {
        Start-Sleep -Milliseconds 500
    }
    if (Get-Service -Name $Name -ErrorAction SilentlyContinue) {
        throw "Service $Name is marked for deletion but still registered after 30 s. Another program (services.msc, Event Viewer, Task Manager) probably holds it open. Close those windows (or reboot) and run this script again."
    }
}

$InstallDir = (Resolve-Path $InstallDir).Path.TrimEnd('\')
$exe = Join-Path $InstallDir "blue-onyx-prism-service.exe"
if (-not (Test-Path $exe)) {
    throw "blue-onyx-prism-service.exe not found in '$InstallDir'. Pass -InstallDir."
}
if (-not (Test-Path (Join-Path $InstallDir "openvino"))) {
    Write-Warning "No 'openvino' folder next to the service exe. Run 'blue-onyx-prism setup-openvino' first."
}
if ($InstallDir -like "$env:SystemDrive\Users\*") {
    Write-Warning "The install dir is under a user profile; its ACL is reset to Administrators/SYSTEM full and Users read-only. Prefer e.g. C:\Program Files\BlueOnyxPrism."
}

$useSystem = $Account -ieq "LocalSystem" -or $Account -ieq "NT AUTHORITY\SYSTEM"
$runAs = if ($Account -and -not $useSystem) { $Account } elseif ($useSystem) { "LocalSystem" } else { $VirtualAccount }

Write-Host "Creating event log source $EventSource ..."
try {
    if (-not [System.Diagnostics.EventLog]::SourceExists($EventSource)) {
        New-EventLog -LogName Application -Source $EventSource
    }
} catch { Write-Warning "Could not create event log source: $_" }

# ServicesPipeTimeout is machine-wide (every service's start timeout). Record the value found
# before the first install (-1 = not set) so uninstall_service.ps1 can put it back, and never
# lower a larger value someone else set.
$ControlKey = "HKLM:\SYSTEM\CurrentControlSet\Control"
$StateKey = "HKLM:\SOFTWARE\BlueOnyxPrism"
$PipeTimeoutMs = 600000
$current = (Get-ItemProperty -Path $ControlKey -Name ServicesPipeTimeout -ErrorAction SilentlyContinue).ServicesPipeTimeout
if (-not (Test-Path $StateKey)) { New-Item -Path $StateKey -Force | Out-Null }
if ($null -eq (Get-ItemProperty -Path $StateKey -Name PreviousServicesPipeTimeout -ErrorAction SilentlyContinue)) {
    $previous = if ($null -eq $current) { -1 } else { [int64]$current }
    New-ItemProperty -Path $StateKey -Name PreviousServicesPipeTimeout -Value $previous -PropertyType QWord -Force | Out-Null
}
if ($null -eq $current -or [int64]$current -lt $PipeTimeoutMs) {
    Write-Host "Setting ServicesPipeTimeout to $PipeTimeoutMs ms (takes effect after reboot; uninstall restores the previous value) ..."
    Set-ItemProperty -Path $ControlKey -Name ServicesPipeTimeout -Value $PipeTimeoutMs -Type DWord
} else {
    Write-Host "ServicesPipeTimeout is already $current ms; left as is."
}

# Pre-rename service (Blue Onyx OpenVINO): remove it so the two don't fight over the port.
if (Get-Service -Name "BlueOnyxOpenVINOService" -ErrorAction SilentlyContinue) {
    Write-Host "Removing the old BlueOnyxOpenVINOService ..."
    Remove-ServiceIfPresent "BlueOnyxOpenVINOService"
}
if (Get-Service -Name $ServiceName -ErrorAction SilentlyContinue) {
    Write-Host "Service already exists; stopping and deleting it first ..."
    Remove-ServiceIfPresent $ServiceName
}

Write-Host "Registering service $ServiceName (account: $runAs) ..."
$binPath = "`"$exe`""
if ($Account -and -not $useSystem) {
    # The password goes to the SCM through a credential object, never on a command line.
    $cred = Get-Credential -UserName $Account -Message "Password for $Account (service logon)"
    New-Service -Name $ServiceName -BinaryPathName $binPath -DisplayName $DisplayName `
        -StartupType Automatic -Credential $cred | Out-Null
    Write-Host "Make sure $Account has the 'Log on as a service' right (secpol.msc > Local Policies > User Rights Assignment)."
} else {
    Invoke-Sc create $ServiceName binPath= $binPath start= auto DisplayName= $DisplayName obj= LocalSystem
    Invoke-Sc sidtype $ServiceName unrestricted
    if (-not $useSystem) {
        # Virtual account: created by the SCM for this service, no password.
        Invoke-Sc config $ServiceName obj= $VirtualAccount
    }
}
Invoke-Sc config $ServiceName type= own
# Service SID in the token. "unrestricted", not "restricted": a restricted token would need an
# explicit grant on every file, registry key and device (GPU driver) the process opens. Set again
# here so the -Account path gets it too.
Invoke-Sc sidtype $ServiceName unrestricted
# Required privileges: the SCM removes every privilege not listed from the service process, so
# even under LocalSystem it runs without SeDebug, SeImpersonate, SeBackup, ... SeChangeNotify
# (bypass traverse checking) is the one the process needs to open files below folders it cannot
# list. If a future Windows/driver needs another one, the service fails to start with an
# access-denied error in the event log: add it here, separated by '/'.
Invoke-Sc privs $ServiceName SeChangeNotifyPrivilege
Invoke-Sc description $ServiceName "Blue Iris / CodeProject.AI compatible object detection (OpenVINO / ONNX Runtime)"
Invoke-Sc failure $ServiceName reset= 86400 actions= restart/5000/restart/5000/restart/5000

# --- Permissions --------------------------------------------------------------------------------
# Well-known SIDs, so the script works on non-English Windows.
$Admins = "*S-1-5-32-544"; $System = "*S-1-5-18"; $Users = "*S-1-5-32-545"; $CreatorOwner = "*S-1-3-0"

Write-Host "Resetting the ACL of $InstallDir (Administrators/SYSTEM full, Users read+execute) ..."
Invoke-Icacls $InstallDir /inheritance:r /grant:r "${Admins}:(OI)(CI)F" "${System}:(OI)(CI)F" "${Users}:(OI)(CI)RX" /Q
# Everything below inherits that ACL again (explicit grants from earlier installs are dropped).
Invoke-Icacls "$InstallDir\*" /reset /T /C /Q

if (-not $useSystem) {
    $svcAcct = $runAs
    # The install dir itself: read, plus create files (WD) and folders (AD), this folder only. With
    # CREATOR OWNER Modify (inherit only), what the service creates (staging dirs, the config's
    # temp file, .installed.json, .downloads.lock, benchmark.json) stays writable for it, while the
    # exe and the shipped files remain read-only.
    Invoke-Icacls $InstallDir /grant "${svcAcct}:(OI)(CI)RX" "${svcAcct}:(WD,AD)" "${CreatorOwner}:(OI)(CI)(IO)M" /Q

    # Data dirs (created when missing) and files: Modify, inherited by everything inside.
    $dataDirs = [System.Collections.Generic.List[string]]::new()
    foreach ($d in @("models", "cache", "tools", "openvino", "onnxruntime", ".downloads")) {
        $dataDirs.Add((Join-Path $InstallDir $d))
    }
    # Paths the service config names (relative ones resolve against the install dir; models_dir
    # against download_dir).
    $configPath = Join-Path $InstallDir $ConfigName
    if (Test-Path $configPath) {
        try {
            $cfg = Get-Content -Raw $configPath | ConvertFrom-Json
            # A key the config omits reads as $null (strict mode would otherwise throw on it).
            $key = { param($n) $m = $cfg.PSObject.Properties[$n]; if ($m) { $m.Value } else { $null } }
            $resolve = {
                param($p, $base)
                if (-not $p) { return $null }
                if ([System.IO.Path]::IsPathRooted($p)) { return $p }
                return (Join-Path $base $p)
            }
            $root = & $resolve (& $key "download_dir") $InstallDir
            if (-not $root) { $root = $InstallDir }
            foreach ($p in @(
                    (& $resolve (& $key "download_dir") $InstallDir),
                    (& $resolve (& $key "models_dir") $root),
                    (& $resolve (& $key "cache_dir") $InstallDir),
                    (& $resolve (& $key "save_image_path") $InstallDir),
                    (& $resolve (& $key "log_path") $InstallDir))) {
                if ($p) { $dataDirs.Add($p) }
            }
        } catch { Write-Warning "Could not read $configPath ($_); grant Modify on its data paths by hand." }
    }
    foreach ($d in ($dataDirs | Select-Object -Unique)) {
        if (-not (Test-Path $d)) { New-Item -ItemType Directory -Force -Path $d | Out-Null }
        Invoke-Icacls $d /grant "${svcAcct}:(OI)(CI)M" /C /Q
    }
    foreach ($f in @($ConfigName, "benchmark.json", ".installed.json", ".downloads.lock")) {
        $p = Join-Path $InstallDir $f
        if (Test-Path $p) { Invoke-Icacls $p /grant "${svcAcct}:M" /Q }
    }
    Write-Host "Granted $svcAcct Modify on: $(($dataDirs | Select-Object -Unique) -join ', ') and the config file."
    Write-Host "A save_image_path or log_path added later needs: icacls <dir> /grant `"${svcAcct}:(OI)(CI)M`""
}

Write-Host "Adding firewall rule for TCP $Port ..."
$ruleName = "Blue Onyx Prism ($Port)"
if (-not (Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue)) {
    New-NetFirewallRule -DisplayName $ruleName -Direction Inbound -Protocol TCP -LocalPort $Port -Action Allow | Out-Null
}

if (-not $NoStart) {
    Write-Host "Starting service ..."
    Start-Service $ServiceName
    Start-Sleep -Seconds 3
    Get-Service $ServiceName | Format-Table -AutoSize
    Write-Host "First start compiles the models; watch http://127.0.0.1:$Port/ and the Application event log."
    Write-Host "If the GPU is not used under $runAs, reinstall with -Account LocalSystem."
}

Write-Host ""
Write-Host "Manage with:  net start $ServiceName | net stop $ServiceName | .\scripts\uninstall_service.ps1"
Write-Host "Config file:  $(Join-Path $InstallDir $ConfigName)"
