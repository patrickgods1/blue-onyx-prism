#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install / run / uninstall smoke test for the Blue Onyx Prism MSI (CI; modifies the machine).
.PARAMETER Msi
    Path of the MSI to test.
.PARAMETER Port
    HTTP port the service listens on (default 32168).
#>
param(
    [Parameter(Mandatory)][string]$Msi,
    [int]$Port = 32168
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Msi = (Resolve-Path $Msi).Path
$Service = "BlueOnyxPrismService"
$DataDir = Join-Path $env:ProgramData "BlueOnyxPrism"
$Config = Join-Path $DataDir "blue_onyx_prism_config_service.json"
$InstallLog = Join-Path (Get-Location) "msi-install.log"
$UninstallLog = Join-Path (Get-Location) "msi-uninstall.log"

function Show-LogTail([string]$Path) {
    if (Test-Path $Path) {
        Write-Host "---- tail of $Path ----"
        Get-Content $Path -Tail 80 -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
        Write-Host "---- end ----"
    }
}

function Fail([string]$Message, [string[]]$Logs = @()) {
    Write-Host "SMOKE TEST FAILED: $Message"
    foreach ($l in $Logs) { Show-LogTail $l }
    Get-WinEvent -FilterHashtable @{ LogName = "Application"; ProviderName = "BlueOnyxPrism" } -MaxEvents 20 -ErrorAction SilentlyContinue |
        ForEach-Object { Write-Host "[event] $($_.TimeCreated) $($_.Message)" }
    exit 1
}

function Run-Msiexec([string[]]$MsiArgs) {
    $p = Start-Process -FilePath msiexec.exe -ArgumentList $MsiArgs -Wait -PassThru
    return $p.ExitCode
}

# --- install ------------------------------------------------------------------------------------
Write-Host "Installing $Msi ..."
$rc = Run-Msiexec @("/i", "`"$Msi`"", "/qn", "/l*v", "`"$InstallLog`"", "PORT=$Port")
if ($rc -ne 0 -and $rc -ne 3010) { Fail "msiexec /i exited with $rc" @($InstallLog) }

$svc = Get-CimInstance Win32_Service -Filter "Name='$Service'"
if (-not $svc) { Fail "service $Service does not exist after install" @($InstallLog) }
if ($svc.StartName -ne "NT AUTHORITY\LocalService") { Fail "service runs as '$($svc.StartName)', expected NT AUTHORITY\LocalService" @($InstallLog) }
Write-Host "Service registered: $($svc.PathName) as $($svc.StartName)"

$deadline = (Get-Date).AddSeconds(60)
while ((Get-Service $Service).Status -ne "Running" -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
if ((Get-Service $Service).Status -ne "Running") { Fail "service is $((Get-Service $Service).Status), expected Running within 60 s" @($InstallLog) }
Write-Host "Service is Running."

# /v1/status answers 200 (ready) or 503 (models still initializing/downloading); only the listener matters.
$url = "http://127.0.0.1:$Port/v1/status"
$deadline = (Get-Date).AddSeconds(120)
$code = 0
while ((Get-Date) -lt $deadline) {
    try {
        $code = [int](Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5).StatusCode
    } catch {
        $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
    }
    if ($code -eq 200 -or $code -eq 503) { break }
    Start-Sleep -Seconds 2
}
if ($code -ne 200 -and $code -ne 503) { Fail "$url did not answer 200/503 within 120 s (last: $code)" @($InstallLog) }
Write-Host "GET $url -> $code"

if (-not (Test-Path $Config)) { Fail "$Config was not created" @($InstallLog) }
Write-Host "Config present: $Config"

if (-not (Get-NetFirewallRule -DisplayName "Blue Onyx Prism ($Port)" -ErrorAction SilentlyContinue)) {
    Fail "firewall rule 'Blue Onyx Prism ($Port)' missing after install" @($InstallLog)
}

# Tray: exe, start-at-login Run value, Start Menu shortcut (the tray itself is not run: no desktop session in CI).
$TrayExe = Join-Path ${env:ProgramFiles} "Blue Onyx Prism\blue-onyx-prism-tray.exe"
$RunKey = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
$StartMenu = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs\Blue Onyx Prism"
if (-not (Test-Path $TrayExe)) { Fail "$TrayExe missing after install" @($InstallLog) }
$run = (Get-ItemProperty -Path $RunKey -Name BlueOnyxPrismTray -ErrorAction SilentlyContinue).BlueOnyxPrismTray
if (-not $run) { Fail "Run value BlueOnyxPrismTray missing after install" @($InstallLog) }
if ($run -ne "`"$TrayExe`"") { Fail "Run value is '$run', expected '`"$TrayExe`"'" @($InstallLog) }
if (-not (Test-Path (Join-Path $StartMenu "Blue Onyx Prism.lnk"))) { Fail "Start Menu shortcut 'Blue Onyx Prism' missing" @($InstallLog) }
if (-not (Get-ChildItem -Path $StartMenu -Filter "Blue Onyx Prism Web UI*" -ErrorAction SilentlyContinue)) { Fail "Start Menu shortcut 'Blue Onyx Prism Web UI' missing" @($InstallLog) }
Write-Host "Tray exe, Run value and Start Menu shortcuts present."

# --- uninstall ----------------------------------------------------------------------------------
Write-Host "Uninstalling ..."
$rc = Run-Msiexec @("/x", "`"$Msi`"", "/qn", "/l*v", "`"$UninstallLog`"")
if ($rc -ne 0 -and $rc -ne 3010) { Fail "msiexec /x exited with $rc" @($UninstallLog) }

$deadline = (Get-Date).AddSeconds(30)
while ((Get-Service $Service -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
if (Get-Service $Service -ErrorAction SilentlyContinue) { Fail "service $Service still registered after uninstall" @($UninstallLog) }
if (Get-NetFirewallRule -DisplayName "Blue Onyx Prism ($Port)" -ErrorAction SilentlyContinue) {
    Fail "firewall rule 'Blue Onyx Prism ($Port)' still present after uninstall" @($UninstallLog)
}
if (Test-Path $TrayExe) { Fail "$TrayExe still present after uninstall" @($UninstallLog) }
if (Get-ItemProperty -Path $RunKey -Name BlueOnyxPrismTray -ErrorAction SilentlyContinue) { Fail "Run value BlueOnyxPrismTray still present after uninstall" @($UninstallLog) }
if (Test-Path $StartMenu) { Fail "Start Menu folder $StartMenu still present after uninstall" @($UninstallLog) }
if (-not (Test-Path $DataDir)) { Fail "$DataDir was removed by the uninstall (must be kept)" @($UninstallLog) }

Write-Host "SMOKE TEST PASSED (service, firewall rule, tray and data folder behave as expected)."
