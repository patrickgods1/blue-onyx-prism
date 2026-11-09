#Requires -RunAsAdministrator
<#
.SYNOPSIS
    Install / run / uninstall smoke test for the Blue Onyx Prism MSI (CI; modifies the machine).
    Installs four times: an uninstall with REMOVEDATA=1 must leave nothing behind; a plain one
    keeps only the data folder, a seeded dataset file included; an install over that data with an
    older version's ACL corrects it, and KEEPDATA=1 (old scripts) keeps it again, even with REMOVEDATA=1; a last install
    over it and REMOVEDATA=1 removes everything, the protected token folder too. Each install also
    checks the data folder's DACL, the auth folder's protected DACL, the service's required
    privileges, both firewall rules' scope and profiles, that the service runs on the Visual
    C++ runtime installed next to it, and the web UI: HTTPS answers behind a sign-in, the plain
    port sends browsers there, and `setup-code` prints the first account's code from the
    protected auth folder. Needs PowerShell 7 (-SkipCertificateCheck, -SkipHttpErrorCheck).
.PARAMETER Msi
    Path of the MSI to test.
.PARAMETER Port
    HTTP port the service listens on (default 32168).
.PARAMETER HttpsPort
    HTTPS port of the web UI, for its firewall rule (default 32169).
#>
param(
    [Parameter(Mandatory)][string]$Msi,
    [int]$Port = 32168,
    [int]$HttpsPort = 32169
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
$AuthDir = Join-Path $DataDir "auth"
$FirewallRules = @("Blue Onyx Prism ($Port)", "Blue Onyx Prism HTTPS ($HttpsPort)")
# Stands for what no download brings back (dataset frames, reviewed labels, training runs). In a
# folder of its own: nothing the service prunes walks it.
$SeedFile = Join-Path $DataDir "dataset\smoke-test\frame.labels.json"
$SeedText = "reviewed labels"
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

# Every call passes /norestart: without it a silent msiexec that finds a file in use may restart the
# machine on its own, which ends the job with no log.
function Run-Msiexec([string[]]$MsiArgs) {
    $p = Start-Process -FilePath msiexec.exe -ArgumentList $MsiArgs -Wait -PassThru
    return $p.ExitCode
}

# 3010 is success with a restart held back: a file was in use. Say so, with the log.
function Note-RebootWanted([int]$Code, [string]$Log) {
    if ($Code -eq 3010) {
        Write-Host "msiexec exited with 3010: it wanted a restart, so a file was in use."
        Show-LogTail $Log
    }
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
    # `sc failureflag 1`: recovery also when the service stops itself with an error code (a start
    # that gave up on a busy port), not only on a crash.
    if (-not $svcKey.PSObject.Properties["FailureActionsOnNonCrashFailures"] -or $svcKey.FailureActionsOnNonCrashFailures -ne 1) {
        Fail "the service's failure actions do not apply to non-crash failures (sc failureflag)" @($Log)
    }
    Write-Host "Failure actions apply to non-crash failures."

    # The HTTP port (Blue Iris) and the HTTPS port (web UI): the same scope and profiles.
    foreach ($name in $FirewallRules) {
        $rules = @(Get-NetFirewallRule -DisplayName $name)
        if ($rules.Count -ne 1) { Fail "$($rules.Count) firewall rules named '$name', expected 1" @($Log) }
        $scope = @(($rules[0] | Get-NetFirewallAddressFilter).RemoteAddress)
        $outside = @($scope | Where-Object { $_ -notlike "LocalSubnet*" })
        if ($scope.Count -eq 0 -or $outside.Count -gt 0) { Fail "firewall rule '$name' remote addresses are '$($scope -join ', ')', expected LocalSubnet" @($Log) }
        $fwProfiles = "$($rules[0].Profile)"
        if ($fwProfiles -ne "Domain, Private") { Fail "firewall rule '$name' profiles are '$fwProfiles', expected 'Domain, Private'" @($Log) }
        Write-Host "Firewall rule '$name': $($scope -join ', '), $fwProfiles."
    }

    # <data root>\auth (HTTPS key): its own protected DACL, nothing for Users.
    if (-not (Test-Path $AuthDir)) { Fail "$AuthDir was not created" @($Log) }
    Assert-ExactDacl $AuthDir @{
        $Administrators = "(OI)(CI)(F)"
        $LocalSystem    = "(OI)(CI)(F)"
        $LocalService   = "(OI)(CI)(M)"
    } @($Log)
    $authSids = @((Get-Acl -LiteralPath $AuthDir).Access | ForEach-Object {
        $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value })
    if ($authSids -contains "S-1-5-32-545") { Fail "$AuthDir has an entry for Users (S-1-5-32-545)" @($Log) }
    Write-Host "Auth folder DACL is protected; Users have no access."
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

# The Visual C++ runtime ships next to the exes (stage-vcruntime.ps1), and the running service
# took those copies, not System32's: Windows looks in the program folder first, for the OpenVINO
# and ONNX Runtime DLLs the service loaded from the data folder too. Ready (200) means those
# loaded and serve, so this checks them against the shipped runtime.
$VcRuntimeDlls = "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"
function Assert-VcRuntime([int]$ProcessId, [string]$Log) {
    foreach ($name in $VcRuntimeDlls) {
        $path = Join-Path $ProgDir $name
        if (-not (Test-Path $path)) { Fail "$path missing after install" @($Log) }
        $v = (Get-Item $path).VersionInfo
        Write-Host "Installed $name $($v.FileVersion)"
    }
    try {
        $modules = @((Get-Process -Id $ProcessId -ErrorAction Stop).Modules)
    } catch {
        Fail "cannot list the modules of the service process $ProcessId`: $($_.Exception.Message)" @($Log)
    }
    # The exe links vcruntime140 itself, so it must be loaded, from the program folder. msvcp140 is
    # loaded only when a C++ runtime DLL links it dynamically (first CI run: the runtime that
    # served did not); when it is loaded it must be the shipped copy too.
    foreach ($name in "vcruntime140.dll", "msvcp140.dll") {
        $loaded = @($modules | Where-Object { $_.ModuleName -ieq $name })
        if ($loaded.Count -eq 0) {
            if ($name -ieq "vcruntime140.dll") {
                Fail "the service has no vcruntime140.dll loaded: the exe no longer links the runtime dynamically, or the module list is incomplete" @($Log)
            }
            Write-Host "$name is not loaded in the service process (no loaded DLL links it dynamically)."
            continue
        }
        foreach ($m in $loaded) {
            if ((Split-Path -Parent $m.FileName) -ine $ProgDir) {
                Fail "the service loaded $($m.FileName), not the copy in $ProgDir" @($Log)
            }
        }
    }
    $runtimes = @($modules | Where-Object { $_.ModuleName -match '^(openvino|onnxruntime|tbb)' } | ForEach-Object { $_.FileName })
    $crt = @($modules | Where-Object { $_.ModuleName -match '^(vcruntime|msvcp|ucrtbase)' } | ForEach-Object { $_.FileName })
    Write-Host "Service uses the shipped VC++ runtime. Runtime DLLs loaded: $($runtimes -join ', '). C runtime DLLs: $($crt -join ', ')"
}

# The web UI: HTTPS behind a sign-in (the service's own self-signed certificate, so no
# certificate check here), the plain port sending browsers there (308), and the first account's
# one-time setup code, which the service made in the auth folder and `setup-code` prints. The code
# itself is never written to the job's log.
function Assert-WebUi([string]$Log) {
    $tls = "https://127.0.0.1:$HttpsPort"
    $html = @{ Accept = "text/html" }
    $deadline = (Get-Date).AddSeconds(60)
    $code = 0
    while ((Get-Date) -lt $deadline) {
        try {
            # The sign-in page, or the setup page it sends to before the first account.
            $code = [int](Invoke-WebRequest -Uri "$tls/login" -Headers $html -SkipCertificateCheck -UseBasicParsing -TimeoutSec 10).StatusCode
        } catch {
            $code = Get-HttpCode $_
        }
        if ($code -eq 200) { break }
        Start-Sleep -Seconds 2
    }
    if ($code -ne 200) { Fail "$tls/login did not answer 200 within 60 s (last: $code)" @($Log) }
    Write-Host "GET $tls/login -> 200"
    $r = Invoke-WebRequest -Uri "$tls/config" -Headers $html -SkipCertificateCheck -MaximumRedirection 0 -SkipHttpErrorCheck -UseBasicParsing -TimeoutSec 10
    if ([int]$r.StatusCode -ne 303) { Fail "$tls/config without a sign-in answered $([int]$r.StatusCode), expected 303 to the login page" @($Log) }
    Write-Host "GET $tls/config without a sign-in -> 303"
    $plain = "http://127.0.0.1:$Port/config"
    $r = Invoke-WebRequest -Uri $plain -Headers $html -MaximumRedirection 0 -SkipHttpErrorCheck -UseBasicParsing -TimeoutSec 10
    $location = "$($r.Headers['Location'])"
    if ([int]$r.StatusCode -ne 308 -or $location -ne "$tls/config") {
        Fail "$plain answered $([int]$r.StatusCode) to '$location', expected 308 to $tls/config" @($Log)
    }
    Write-Host "GET $plain -> 308 $location"

    $codeFile = Join-Path $AuthDir "setup-code"
    if (-not (Test-Path -LiteralPath $codeFile)) { Fail "$codeFile missing: the service made no setup code" @($Log) }
    $codeSids = @((Get-Acl -LiteralPath $codeFile).Access | ForEach-Object {
        $_.IdentityReference.Translate([System.Security.Principal.SecurityIdentifier]).Value })
    if ($codeSids -contains "S-1-5-32-545") { Fail "$codeFile has an entry for Users (S-1-5-32-545)" @($Log) }
    $cli = Join-Path $ProgDir "blue-onyx-prism.exe"
    $out = @(& $cli --config $Config setup-code)
    if ($LASTEXITCODE -ne 0) { Fail "setup-code exited with $LASTEXITCODE" @($Log) }
    if (-not (($out -join "`n") -match '(?m)^Setup code: [A-Z2-7-]{20,}')) { Fail "setup-code printed no setup code" @($Log) }
    Write-Host "setup-code prints a code; $codeFile has no entry for Users."
}

function Install-AndCheck([string]$Log) {
    Write-Host "Installing $Msi ..."
    $rc = Run-Msiexec @("/i", "`"$Msi`"", "/qn", "/norestart", "/l*v", "`"$Log`"", "PORT=$Port", "HTTPS_PORT=$HttpsPort")
    if ($rc -ne 0 -and $rc -ne 3010) { Fail "msiexec /i exited with $rc" @($Log) }
    Note-RebootWanted $rc $Log

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
    Assert-VcRuntime $pidNow $Log

    if (-not (Test-Path $Config)) { Fail "$Config was not created" @($Log) }
    Write-Host "Config present: $Config"

    foreach ($name in $FirewallRules) {
        if (-not (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue)) {
            Fail "firewall rule '$name' missing after install" @($Log)
        }
    }
    Assert-Hardening $Log
    Assert-WebUi $Log

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

function Set-SeedFile {
    New-Item -ItemType Directory -Force -Path (Split-Path -Parent $SeedFile) | Out-Null
    Set-Content -Path $SeedFile -Value $SeedText
}

function Assert-DataKept([string]$Log) {
    if (-not (Test-Path $Config)) { Fail "$Config was removed" @($Log) }
    if (-not (Test-Path $SeedFile)) { Fail "$SeedFile was removed" @($Log) }
    if ((Get-Content -Path $SeedFile -Raw).Trim() -cne $SeedText) { Fail "$SeedFile was changed" @($Log) }
    Write-Host "Data folder kept: config and the seeded dataset file are there."
}

# $Properties: what is passed to msiexec. Only REMOVEDATA=1 may delete the data folder.
function Uninstall-AndCheck([string]$Log, [string[]]$Properties = @()) {
    Write-Host "Uninstalling ($($Properties -join ' ')) ..."
    $rc = Run-Msiexec (@("/x", "`"$Msi`"", "/qn", "/norestart", "/l*v", "`"$Log`"") + $Properties)
    if ($rc -ne 0 -and $rc -ne 3010) { Fail "msiexec /x exited with $rc" @($Log) }
    Note-RebootWanted $rc $Log

    $deadline = (Get-Date).AddSeconds(30)
    while ((Get-Service $Service -ErrorAction SilentlyContinue) -and (Get-Date) -lt $deadline) { Start-Sleep -Seconds 1 }
    if (Get-Service $Service -ErrorAction SilentlyContinue) { Fail "service $Service still registered after uninstall" @($Log) }
    foreach ($name in $FirewallRules) {
        if (Get-NetFirewallRule -DisplayName $name -ErrorAction SilentlyContinue) {
            Fail "firewall rule '$name' still present after uninstall" @($Log)
        }
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
    if ($Properties -contains "KEEPDATA=1" -or $Properties -notcontains "REMOVEDATA=1") {
        Assert-DataKept $Log
    } elseif (Test-Path $DataDir) {
        $left = (Get-ChildItem -Path $DataDir -Recurse -Force | Select-Object -First 20 | ForEach-Object { $_.FullName }) -join ", "
        Fail "$DataDir still present after uninstall ($left)" @($Log)
    }
}

# 1. REMOVEDATA=1: nothing may be left behind.
Install-AndCheck $InstallLog
Uninstall-AndCheck $UninstallLog @("REMOVEDATA=1")
# 2. Over the deleted folder, then a plain uninstall, as from Settings > Apps: only the data stays.
Install-AndCheck (Join-Path (Get-Location) "msi-install-2.log")
Set-SeedFile
Uninstall-AndCheck (Join-Path (Get-Location) "msi-uninstall-default.log")
# 3. Over that data, with an older version's ACL: the install corrects it and keeps the data. Then
# what an old script passes, and a REMOVEDATA that is not 1: both keep it.
Set-OlderVersionData
$Log3 = Join-Path (Get-Location) "msi-install-3.log"
Install-AndCheck $Log3
Assert-DataKept $Log3
Assert-TokenFolderProtected $Log3
Uninstall-AndCheck (Join-Path (Get-Location) "msi-uninstall-keepdata.log") @("KEEPDATA=1", "REMOVEDATA=1")
# 4. REMOVEDATA=1 also removes the protected token folder.
Install-AndCheck (Join-Path (Get-Location) "msi-install-4.log")
Uninstall-AndCheck (Join-Path (Get-Location) "msi-uninstall-removedata.log") @("REMOVEDATA=1")

Write-Host "SMOKE TEST PASSED (a plain uninstall and KEEPDATA=1 keep only the data folder; a reinstall keeps its data and corrects an older ACL; REMOVEDATA=1 leaves nothing behind)."
