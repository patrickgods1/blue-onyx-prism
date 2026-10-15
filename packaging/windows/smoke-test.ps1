#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install / run / uninstall smoke test for the Blue Onyx Prism MSI (CI; modifies the machine).
    Installs twice: a plain uninstall must leave nothing behind; one with KEEPDATA=1 keeps only
    the data folder.
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

$ProgDir = Join-Path ${env:ProgramFiles} "Blue Onyx Prism"
$TrayExe = Join-Path $ProgDir "blue-onyx-prism-tray.exe"
$RunKey = "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Run"
$StartMenu = Join-Path $env:ProgramData "Microsoft\Windows\Start Menu\Programs\Blue Onyx Prism"
$ProductKey = "HKLM:\SOFTWARE\BlueOnyxPrism"
$EventSourceKey = "HKLM:\SYSTEM\CurrentControlSet\Services\EventLog\Application\BlueOnyxPrism"

function Install-AndCheck([string]$Log) {
    Write-Host "Installing $Msi ..."
    $rc = Run-Msiexec @("/i", "`"$Msi`"", "/qn", "/l*v", "`"$Log`"", "PORT=$Port")
    if ($rc -ne 0 -and $rc -ne 3010) { Fail "msiexec /i exited with $rc" @($Log) }

    $svc = Get-CimInstance Win32_Service -Filter "Name='$Service'"
    if (-not $svc) { Fail "service $Service does not exist after install" @($Log) }
    if ($svc.StartName -ne "NT AUTHORITY\LocalService") { Fail "service runs as '$($svc.StartName)', expected NT AUTHORITY\LocalService" @($Log) }
    Write-Host "Service registered: $($svc.PathName) as $($svc.StartName)"

    $deadline = (Get-Date).AddSeconds(60)
    while ((Get-Service $Service).Status -ne "Running" -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 2 }
    if ((Get-Service $Service).Status -ne "Running") { Fail "service is $((Get-Service $Service).Status), expected Running within 60 s" @($Log) }
    Write-Host "Service is Running."
    $pidRunning = (Get-CimInstance Win32_Service -Filter "Name='$Service'").ProcessId

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
    if ($code -ne 200 -and $code -ne 503) { Fail "$url did not answer 200/503 within 120 s (last: $code)" @($Log) }
    Write-Host "GET $url -> $code"

    # Ready (200) means the downloaded runtime loads and the model serves; answering alone could
    # hide a service that fails and is restarted by its failure actions.
    $deadline = (Get-Date).AddSeconds(360)
    while ($code -ne 200 -and (Get-Date) -lt $deadline) {
        Start-Sleep -Seconds 2
        try {
            $code = [int](Invoke-WebRequest -Uri $url -UseBasicParsing -TimeoutSec 5).StatusCode
        } catch {
            $code = if ($_.Exception.Response) { [int]$_.Exception.Response.StatusCode } else { 0 }
        }
    }
    if ($code -ne 200) { Fail "$url not ready (200) within 6 min (last: $code)" @($Log) }
    $pidNow = (Get-CimInstance Win32_Service -Filter "Name='$Service'").ProcessId
    if ($pidNow -ne $pidRunning) { Fail "the service process changed ($pidRunning -> $pidNow): it was restarted" @($Log) }
    Write-Host "Service ready, same process."

    if (-not (Test-Path $Config)) { Fail "$Config was not created" @($Log) }
    Write-Host "Config present: $Config"

    if (-not (Get-NetFirewallRule -DisplayName "Blue Onyx Prism ($Port)" -ErrorAction SilentlyContinue)) {
        Fail "firewall rule 'Blue Onyx Prism ($Port)' missing after install" @($Log)
    }

    # Tray: exe, start-at-login Run value, Start Menu shortcut (the tray itself is not run: no desktop session in CI).
    if (-not (Test-Path $TrayExe)) { Fail "$TrayExe missing after install" @($Log) }
    $run = (Get-ItemProperty -Path $RunKey -Name BlueOnyxPrismTray -ErrorAction SilentlyContinue).BlueOnyxPrismTray
    if (-not $run) { Fail "Run value BlueOnyxPrismTray missing after install" @($Log) }
    if ($run -ne "`"$TrayExe`"") { Fail "Run value is '$run', expected '`"$TrayExe`"'" @($Log) }
    if (-not (Test-Path (Join-Path $StartMenu "Blue Onyx Prism.lnk"))) { Fail "Start Menu shortcut 'Blue Onyx Prism' missing" @($Log) }
    if (-not (Get-ChildItem -Path $StartMenu -Filter "Blue Onyx Prism Web UI*" -ErrorAction SilentlyContinue)) { Fail "Start Menu shortcut 'Blue Onyx Prism Web UI' missing" @($Log) }
    if (-not (Test-Path $ProductKey)) { Fail "$ProductKey missing after install" @($Log) }
    Write-Host "Tray exe, Run value, Start Menu shortcuts and registry key present."

    # What the console exe creates when run from the program folder (config, downloads): not
    # MSI files, so only the uninstall's folder cleanup removes them.
    New-Item -ItemType Directory -Force -Path (Join-Path $ProgDir "openvino") | Out-Null
    Set-Content -Path (Join-Path $ProgDir "openvino\placeholder.dll") -Value "x"
    Set-Content -Path (Join-Path $ProgDir "blue_onyx_prism_config.json") -Value "{}"
}

function Uninstall-AndCheck([string]$Log, [bool]$KeepData) {
    Write-Host "Uninstalling (KEEPDATA=$KeepData) ..."
    $msiArgs = @("/x", "`"$Msi`"", "/qn", "/l*v", "`"$Log`"")
    if ($KeepData) { $msiArgs += "KEEPDATA=1" }
    $rc = Run-Msiexec $msiArgs
    if ($rc -ne 0 -and $rc -ne 3010) { Fail "msiexec /x exited with $rc" @($Log) }

    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Service $Service -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
    if (Get-Service $Service -ErrorAction SilentlyContinue) { Fail "service $Service still registered after uninstall" @($Log) }
    if (Get-NetFirewallRule -DisplayName "Blue Onyx Prism ($Port)" -ErrorAction SilentlyContinue) {
        Fail "firewall rule 'Blue Onyx Prism ($Port)' still present after uninstall" @($Log)
    }
    if (Test-Path $ProgDir) {
        $left = (Get-ChildItem -Path $ProgDir -Recurse -Force | ForEach-Object { $_.FullName }) -join ", "
        Fail "$ProgDir still present after uninstall ($left)" @($Log)
    }
    if (Get-ItemProperty -Path $RunKey -Name BlueOnyxPrismTray -ErrorAction SilentlyContinue) { Fail "Run value BlueOnyxPrismTray still present after uninstall" @($Log) }
    if (Test-Path $StartMenu) { Fail "Start Menu folder $StartMenu still present after uninstall" @($Log) }
    if (Test-Path $ProductKey) { Fail "$ProductKey still present after uninstall" @($Log) }
    if (Test-Path $EventSourceKey) { Fail "event log source key $EventSourceKey still present after uninstall" @($Log) }
    $arp = Get-ItemProperty "HKLM:\SOFTWARE\Microsoft\Windows\CurrentVersion\Uninstall\*", "HKLM:\SOFTWARE\WOW6432Node\Microsoft\Windows\CurrentVersion\Uninstall\*" -ErrorAction SilentlyContinue |
        Where-Object { $_.PSObject.Properties["DisplayName"] -and $_.DisplayName -like "Blue Onyx Prism*" }
    if ($arp) { Fail "Apps & features entry still present after uninstall" @($Log) }
    if ($KeepData) {
        if (-not (Test-Path $Config)) { Fail "$Config was removed by an uninstall with KEEPDATA=1" @($Log) }
        Write-Host "Data folder kept with KEEPDATA=1."
    } elseif (Test-Path $DataDir) {
        $left = (Get-ChildItem -Path $DataDir -Recurse -Force | Select-Object -First 20 | ForEach-Object { $_.FullName }) -join ", "
        Fail "$DataDir still present after uninstall ($left)" @($Log)
    }
}

# 1. Plain uninstall: nothing may be left behind.
Install-AndCheck $InstallLog
Uninstall-AndCheck $UninstallLog $false
# 2. KEEPDATA=1: only the data folder stays.
Install-AndCheck (Join-Path (Get-Location) "msi-install-2.log")
Uninstall-AndCheck (Join-Path (Get-Location) "msi-uninstall-keepdata.log") $true
Remove-Item -Recurse -Force $DataDir

Write-Host "SMOKE TEST PASSED (a full uninstall leaves nothing behind; KEEPDATA=1 keeps only the data folder)."
