#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install / run / uninstall smoke test for the Blue Onyx Prism MSI (CI; modifies the machine).
    Installs three times: a plain uninstall must leave nothing behind; one with KEEPDATA=1 keeps
    only the data folder; an install over data with an older version's ACL corrects it. Each
    install also checks the data folder's DACL, the service's required privileges and the firewall
    rule's scope and profiles.
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

# The HTTP status of a failed Invoke-WebRequest, 0 when there was none (refused, timed out,
# connection closed). Under StrictMode a missing .Response property would throw instead.
function Get-HttpCode($err) {
    $r = $err.Exception.PSObject.Properties["Response"]
    if ($r -and $r.Value) { return [int]$r.Value.StatusCode }
    return 0
}
$Msi = (Resolve-Path $Msi).Path
$Service = "BlueOnyxPrismService"
$DataDir = Join-Path $env:ProgramData "BlueOnyxPrism"
$Config = Join-Path $DataDir "blue_onyx_prism_config_service.json"
$TokenDir = Join-Path $DataDir "tools\hf"
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

# Account names as icacls prints them (translated on a non-English Windows).
function Get-AccountName([string]$Sid) {
    return (New-Object System.Security.Principal.SecurityIdentifier($Sid)).Translate([System.Security.Principal.NTAccount]).Value
}
$Administrators = Get-AccountName "S-1-5-32-544"
$LocalSystem = Get-AccountName "S-1-5-18"
$LocalService = Get-AccountName "S-1-5-19"
$BuiltinUsers = Get-AccountName "S-1-5-32-545"

# The ACEs of $Path as icacls prints them: `<account>:(OI)(CI)(RX)`, the first one after the path.
function Get-Aces([string]$Path, [string[]]$Logs) {
    $out = @(& icacls.exe $Path)
    if ($LASTEXITCODE -ne 0) { Fail "icacls '$Path' exited with $LASTEXITCODE`: $out" $Logs }
    Write-Host ($out -join "`n")
    $aces = @()
    foreach ($line in $out) {
        $text = "$line"
        if ($text.StartsWith($Path)) { $text = $text.Substring($Path.Length) }
        if ($text.Trim() -match '^(.+?):((?:\([A-Za-z,]+\))+)$') {
            $aces += [pscustomobject]@{ Account = $Matches[1]; Rights = $Matches[2] }
        }
    }
    if ($aces.Count -eq 0) { Fail "no ACE could be read from icacls '$Path': $out" $Logs }
    return $aces
}

# Every right in a `(OI)(CI)(RX)` string that lets an account change something. An allow-list of
# what does not: inheritance flags, read, execute, list, read attributes, synchronize.
function Get-WriteRights([string]$Rights) {
    $harmless = "OI", "CI", "IO", "NP", "I", "RX", "R", "GR", "GE", "RD", "REA", "RA", "RC", "X", "S"
    return @($Rights -split '[(),]' | Where-Object { $_ -and $harmless -notcontains $_ })
}

function Assert-NoUserWrite([string]$Path, [string[]]$Logs) {
    foreach ($ace in @(Get-Aces $Path $Logs | Where-Object { $_.Account -eq $BuiltinUsers })) {
        $write = @(Get-WriteRights $ace.Rights)
        if ($write.Count -gt 0) { Fail "$BuiltinUsers can write to $Path`: $($ace.Rights)" $Logs }
    }
}

# $Expected: account -> rights. Exactly these ACEs, none inherited: the DACL is protected.
function Assert-ExactDacl([string]$Path, [hashtable]$Expected, [string[]]$Logs) {
    $aces = @(Get-Aces $Path $Logs)
    foreach ($ace in $aces) {
        if ($ace.Rights -match '\(I\)') { Fail "$Path has an inherited ACE: $($ace.Account):$($ace.Rights)" $Logs }
        if ($Expected[$ace.Account] -cne $ace.Rights) {
            Fail "$Path has $($ace.Account):$($ace.Rights), expected '$($Expected[$ace.Account])'" $Logs
        }
    }
    if ($aces.Count -ne $Expected.Count) { Fail "$Path has $($aces.Count) ACEs, expected $($Expected.Count)" $Logs }
}

function Assert-Hardening([string]$Log) {
    Assert-ExactDacl $DataDir @{
        $Administrators = "(OI)(CI)(F)"
        $LocalSystem    = "(OI)(CI)(F)"
        $LocalService   = "(OI)(CI)(M)"
        $BuiltinUsers   = "(OI)(CI)(RX)"
    } @($Log)
    # What the service created inherits that, on an upgrade too.
    Assert-NoUserWrite $Config @($Log)
    Write-Host "Data folder DACL is protected; Users can only read."

    $svcKey = Get-ItemProperty -Path "HKLM:\SYSTEM\CurrentControlSet\Services\$Service"
    if (-not $svcKey.PSObject.Properties["RequiredPrivileges"]) { Fail "the service has no RequiredPrivileges value" @($Log) }
    $privileges = @($svcKey.RequiredPrivileges)
    if ($privileges.Count -ne 1 -or $privileges[0] -ne "SeChangeNotifyPrivilege") {
        Fail "required privileges are '$($privileges -join ', ')', expected only SeChangeNotifyPrivilege" @($Log)
    }
    Write-Host "Required privileges: $($privileges -join ', ')."

    $rules = @(Get-NetFirewallRule -DisplayName "Blue Onyx Prism ($Port)")
    if ($rules.Count -ne 1) { Fail "$($rules.Count) firewall rules named 'Blue Onyx Prism ($Port)', expected 1" @($Log) }
    $scope = @(($rules[0] | Get-NetFirewallAddressFilter).RemoteAddress)
    $outside = @($scope | Where-Object { $_ -notlike "LocalSubnet*" })
    if ($scope.Count -eq 0 -or $outside.Count -gt 0) { Fail "firewall rule remote addresses are '$($scope -join ', ')', expected LocalSubnet" @($Log) }
    $fwProfiles = "$($rules[0].Profile)"
    if ($fwProfiles -ne "Domain, Private") { Fail "firewall rule profiles are '$fwProfiles', expected 'Domain, Private'" @($Log) }
    Write-Host "Firewall rule: $($scope -join ', '), $fwProfiles."
}

# What an older version left: the data folder inherits from ProgramData, Users can write, and a
# saved token sits in a folder with no ACL of its own.
function Set-OlderVersionData {
    $out = @(& icacls.exe $DataDir /inheritance:e /grant "*S-1-5-32-545:(OI)(CI)M" /Q)
    if ($LASTEXITCODE -ne 0) { Fail "icacls '$DataDir' exited with $LASTEXITCODE`: $out" }
    New-Item -ItemType Directory -Force -Path $TokenDir | Out-Null
    Set-Content -Path (Join-Path $TokenDir "token") -Value "not-a-real-token"
}

function Assert-TokenFolderProtected([string]$Log) {
    Assert-ExactDacl $TokenDir @{
        $Administrators = "(OI)(CI)(F)"
        $LocalSystem    = "(OI)(CI)(F)"
        $LocalService   = "(OI)(CI)(RX)"
    } @($Log)
    $token = Join-Path $TokenDir "token"
    if (@(Get-Aces $token @($Log) | Where-Object { $_.Account -eq $BuiltinUsers }).Count -gt 0) {
        Fail "$BuiltinUsers can read $token" @($Log)
    }
    Write-Host "Token folder DACL is protected; Users have no access."
}

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
            $code = Get-HttpCode $_
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
            $code = Get-HttpCode $_
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
    Assert-Hardening $Log

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
# 3. Over that data, with an older version's ACL: the install corrects it, and the full uninstall
# still removes the protected token folder.
Set-OlderVersionData
$Log3 = Join-Path (Get-Location) "msi-install-3.log"
Install-AndCheck $Log3
Assert-TokenFolderProtected $Log3
Uninstall-AndCheck (Join-Path (Get-Location) "msi-uninstall-3.log") $false

Write-Host "SMOKE TEST PASSED (a full uninstall leaves nothing behind; KEEPDATA=1 keeps only the data folder; an older ACL is corrected)."
