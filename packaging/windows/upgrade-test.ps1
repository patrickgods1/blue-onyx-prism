#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Upgrade test for the MSI (CI only; modifies the machine). Installs the published OldMsi, seeds
    what a user has (a config edit, a reviewed label file and its frame, a training runs.json),
    upgrades to NewMsi and checks: the service runs the new exe, answers a detection, every seeded
    file and the config edit are unchanged, and the Visual C++ runtime the new MSI ships is in the
    program folder. Ends with an uninstall with REMOVEDATA=1. Called by packaging/upgrade-test.sh.
.PARAMETER OldMsi
    The published MSI to upgrade from.
.PARAMETER NewMsi
    The MSI to upgrade to (a higher version: an MSI of the same version is no upgrade).
.PARAMETER NewServiceExe
    The blue-onyx-prism-service.exe NewMsi was built from (to check the upgrade replaced it).
.PARAMETER RepoRoot
    Repository root (test image and runs.json fixture under tests/data).
.PARAMETER LogDir
    Folder for the msiexec logs (default: the current folder).
#>
param(
    [Parameter(Mandatory)][string]$OldMsi,
    [Parameter(Mandatory)][string]$NewMsi,
    [Parameter(Mandatory)][string]$NewServiceExe,
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot)),
    [int]$Port = 32168,
    # Where the msiexec logs go (the workflow uploads dist/*.log).
    [string]$LogDir = (Get-Location).Path
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$OldMsi = (Resolve-Path $OldMsi).Path
$NewMsi = (Resolve-Path $NewMsi).Path
$NewServiceExe = (Resolve-Path $NewServiceExe).Path
$Service = "BlueOnyxPrismService"
$ProgDir = Join-Path ${env:ProgramFiles} "Blue Onyx Prism"
$DataDir = Join-Path $env:ProgramData "BlueOnyxPrism"
$Config = Join-Path $DataDir "blue_onyx_prism_config_service.json"
$Labels = Join-Path $DataDir "dataset\upgrade-test\frame.labels.json"
$Frame = Join-Path $DataDir "dataset\upgrade-test\frame.jpg"
$Runs = Join-Path $DataDir "training\runs.json"
$Image = Join-Path $RepoRoot "tests\data\dog_bike_car.jpg"
$RunsFixture = Join-Path $RepoRoot "tests\data\runs-v0-with-history.json"
$TimeoutEdit = 23 # request_timeout_secs: not the default (15), so a reset to defaults shows
$Url = "http://127.0.0.1:$Port"
$Logs = @()

function Fail([string]$Message) {
    Write-Host "UPGRADE TEST FAILED: $Message"
    foreach ($l in $Logs) {
        if (Test-Path $l) {
            Write-Host "---- tail of $l ----"
            Get-Content $l -Tail 60 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
        }
    }
    Get-WinEvent -FilterHashtable @{ LogName = "Application"; ProviderName = "BlueOnyxPrism" } -MaxEvents 20 -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "[event] $($_.TimeCreated) $($_.Message)" }
    exit 1
}

function Get-HttpCode($err) {
    $r = $err.Exception.PSObject.Properties["Response"]
    if ($r -and $r.Value) { return [int]$r.Value.StatusCode }
    return 0
}

function Wait-Ready([string]$When) {
    $deadline = (Get-Date).AddMinutes(8)
    $code = 0
    while ((Get-Date) -lt $deadline) {
        try {
            $code = [int](Invoke-WebRequest -Uri "$Url/v1/status" -UseBasicParsing -TimeoutSec 5).StatusCode
        } catch {
            $code = Get-HttpCode $_
        }
        if ($code -eq 200) { Write-Host "/v1/status 200 ($When)"; return }
        Start-Sleep -Seconds 2
    }
    Fail "/v1/status not ready (200) within 8 min $When (last: $code)"
}

function Install-Msi([string]$Msi, [string]$Log) {
    $script:Logs += $Log
    Write-Host "Installing $Msi ..."
    $p = Start-Process -FilePath msiexec.exe -Wait -PassThru -ArgumentList @("/i", "`"$Msi`"", "/qn", "/norestart", "/l*v", "`"$Log`"", "PORT=$Port")
    if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { Fail "msiexec /i $Msi exited with $($p.ExitCode)" }
    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Service $Service -ErrorAction SilentlyContinue).Status -ne "Running" -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
    if ((Get-Service $Service -ErrorAction SilentlyContinue).Status -ne "Running") { Fail "service not Running within 60 s after installing $Msi" }
}

function Get-ServicePid { return (Get-CimInstance Win32_Service -Filter "Name='$Service'").ProcessId }

function Get-TimeoutValue {
    $m = [regex]::Match((Get-Content -LiteralPath $Config -Raw), '"request_timeout_secs":\s*(\d+)')
    if ($m.Success) { return [int]$m.Groups[1].Value }
    return $null
}

function Get-Sha([string]$Path) { return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash }

function Get-Products {
    return @(Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*", "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*" -ErrorAction SilentlyContinue |
        Where-Object { $_.PSObject.Properties["DisplayName"] -and $_.DisplayName -like "Blue Onyx Prism*" })
}

$here = $LogDir
Install-Msi $OldMsi (Join-Path $here "msi-upgrade-old.log")
Write-Host "Installed: $((Get-Products | ForEach-Object { "$($_.DisplayName) $($_.DisplayVersion)" }) -join ', ')"
Wait-Ready "on the old version"

Write-Host "== seed a user's data (the service stopped)"
Stop-Service $Service
if ($null -eq (Get-TimeoutValue)) { Fail "$Config has no request_timeout_secs to edit (the old service writes every key on its first start)" }
$text = [regex]::Replace((Get-Content -LiteralPath $Config -Raw), '"request_timeout_secs":\s*\d+', "`"request_timeout_secs`": $TimeoutEdit")
[System.IO.File]::WriteAllText($Config, $text)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $Labels), (Split-Path -Parent $Runs) | Out-Null
Copy-Item -LiteralPath $Image -Destination $Frame
Copy-Item -LiteralPath $RunsFixture -Destination $Runs
Set-Content -LiteralPath $Labels -Value '{"version":1,"status":"reviewed","boxes":[{"label":"vehicle","bbox":[10,20,110,220],"state":"accepted"}],"note":"upgrade test"}'
Start-Service $Service
Wait-Ready "on the old version, with the seeded data"
if ((Get-TimeoutValue) -ne $TimeoutEdit) { Fail "the old version dropped the config edit before the upgrade (request_timeout_secs is '$(Get-TimeoutValue)')" }
$before = @{}
foreach ($f in $Labels, $Frame, $Runs) {
    if (-not (Test-Path -LiteralPath $f)) { Fail "the old version removed the seeded $f before the upgrade" }
    $before[$f] = Get-Sha $f
}
$oldPid = Get-ServicePid

Write-Host "== upgrade to $NewMsi"
Install-Msi $NewMsi (Join-Path $here "msi-upgrade-new.log")
$products = @(Get-Products)
Write-Host "Installed: $(($products | ForEach-Object { "$($_.DisplayName) $($_.DisplayVersion)" }) -join ', ')"
if ($products.Count -ne 1) { Fail "$($products.Count) Blue Onyx Prism products installed after the upgrade, expected 1 (the old one was not removed)" }
$exe = Join-Path $ProgDir "blue-onyx-prism-service.exe"
if ((Get-Sha $exe) -ne (Get-Sha $NewServiceExe)) { Fail "$exe is not the new service exe $NewServiceExe" }
foreach ($name in "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll") {
    if (-not (Test-Path (Join-Path $ProgDir $name))) { Fail "$name missing from $ProgDir after the upgrade" }
}
Wait-Ready "after the upgrade"
$newPid = Get-ServicePid
if ($newPid -eq $oldPid) { Fail "the service process is still $oldPid: the upgrade did not restart it" }
Start-Sleep -Seconds 5
if ((Get-ServicePid) -ne $newPid) { Fail "the service process changed again after the upgrade ($newPid -> $(Get-ServicePid)): it was restarted" }
Write-Host "Service up on the new exe (pid $oldPid -> $newPid)."

Write-Host "== a detection"
$out = & curl.exe -s --max-time 60 -F "image=@$Image" -F "min_confidence=0.2" "$Url/v1/vision/detection"
Write-Host "detection: $out"
if ("$out" -notmatch '"success":true') { Fail "the detection did not succeed" }
if ("$out" -notmatch '"label":"vehicle"') { Fail "the detection found no vehicle on the test image" }

Write-Host "== the user's data"
if ((Get-TimeoutValue) -ne $TimeoutEdit) { Fail "the config edit is gone: request_timeout_secs is '$(Get-TimeoutValue)', was $TimeoutEdit" }
foreach ($f in $Labels, $Frame, $Runs) {
    if (-not (Test-Path -LiteralPath $f)) { Fail "$f is gone after the upgrade" }
    if ((Get-Sha $f) -ne $before[$f]) { Fail "$f changed over the upgrade" }
}
$aside = @(Get-ChildItem -LiteralPath (Split-Path -Parent $Runs) -Filter "*corrupt*" -ErrorAction SilentlyContinue)
if ($aside.Count -gt 0) { Fail "the new version set runs.json aside as damaged: $($aside.Name -join ', ')" }
Write-Host "Config edit kept; label file, frame and runs.json unchanged."

Write-Host "== uninstall (REMOVEDATA=1)"
$log = Join-Path $here "msi-upgrade-uninstall.log"
$script:Logs += $log
$p = Start-Process -FilePath msiexec.exe -Wait -PassThru -ArgumentList @("/x", "`"$NewMsi`"", "/qn", "/norestart", "/l*v", "`"$log`"", "REMOVEDATA=1")
if ($p.ExitCode -ne 0 -and $p.ExitCode -ne 3010) { Fail "msiexec /x exited with $($p.ExitCode)" }
Write-Host "UPGRADE TEST PASSED ($(Split-Path -Leaf $OldMsi) -> $(Split-Path -Leaf $NewMsi))"
