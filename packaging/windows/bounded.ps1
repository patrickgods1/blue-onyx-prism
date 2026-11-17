<#
.SYNOPSIS
    Helpers of the Windows installer tests (smoke-test.ps1, upgrade-test.ps1, diagnose.ps1),
    dot-sourced: external programs run with a time limit, and what the machine looks like when one
    hangs or a check fails.
.DESCRIPTION
    Every program these tests start goes through Invoke-Bounded: stdin closed (nothing waits on the
    runner's console), a limit in seconds, and on a hang the diagnostics, then the program and its
    child processes are killed and the caller fails naming it. `Start-Process -Wait` is not used:
    it has no limit, and PowerShell 7 waits there for every descendant of the process too.
#>

# One command line of $Arguments, each quoted as CommandLineToArgvW (and msiexec) reads it back.
# (ProcessStartInfo.ArgumentList would do it, but only from .NET Core 2.1, PowerShell 6.1.)
function Join-CommandLine([string[]]$Arguments) {
    $parts = foreach ($a in $Arguments) {
        if ($a -ne "" -and $a -notmatch '[\s"]') { $a; continue }
        # Backslashes before a quote, and at the end (before the closing quote), are doubled.
        $q = [regex]::Replace($a, '(\\*)"', { param($m) $m.Groups[1].Value * 2 + '\"' })
        $q = [regex]::Replace($q, '(\\+)$', { param($m) $m.Groups[1].Value * 2 })
        '"' + $q + '"'
    }
    return ($parts -join " ")
}

# Run $File with $Arguments (each one argument, quoted as needed) for at most $Seconds. Returns
# ExitCode (null after a hang), TimedOut, and with -Capture the Output (stdout) and Errors (stderr)
# lines. A program still running at the limit gets Show-Diagnostics (unless -NoDiagnostics), then
# is killed with its process tree.
function Invoke-Bounded {
    param(
        [Parameter(Mandatory)][string]$What,
        [Parameter(Mandatory)][string]$File,
        [string[]]$Arguments = @(),
        [Parameter(Mandatory)][int]$Seconds,
        [string[]]$Logs = @(),
        [switch]$Capture,
        [switch]$NoDiagnostics
    )
    $psi = New-Object System.Diagnostics.ProcessStartInfo $File
    $psi.Arguments = Join-CommandLine $Arguments
    $psi.UseShellExecute = $false
    $psi.RedirectStandardInput = $true
    $psi.RedirectStandardOutput = [bool]$Capture
    $psi.RedirectStandardError = [bool]$Capture
    $started = Get-Date
    $p = [System.Diagnostics.Process]::Start($psi)
    $p.StandardInput.Close()
    $outTask = $null
    $errTask = $null
    if ($Capture) {
        $outTask = $p.StandardOutput.ReadToEndAsync()
        $errTask = $p.StandardError.ReadToEndAsync()
    }
    $timedOut = -not $p.WaitForExit($Seconds * 1000)
    if ($timedOut) {
        Write-Host "HUNG: $What is still running after $Seconds s (pid $($p.Id), started $($started.ToString('o')))."
        if (-not $NoDiagnostics) { Show-Diagnostics -Logs $Logs }
        Write-Host "Killing $What (pid $($p.Id)) and its child processes."
        Stop-Tree $p

    }
    # A grandchild that inherited the pipes keeps them open after the program exits: never wait for
    # their end without a limit.
    $out = @()
    $err = @()
    if ($Capture) {
        if ($outTask.Wait(10000)) { $out = @($outTask.Result -split "`r?`n" | Where-Object { $_ -ne "" }) }
        if ($errTask.Wait(10000)) { $err = @($errTask.Result -split "`r?`n" | Where-Object { $_ -ne "" }) }
    }
    [pscustomobject]@{
        ExitCode = $(if ($timedOut) { $null } else { $p.ExitCode })
        TimedOut = $timedOut
        Output   = $out
        Errors   = $err
    }
}

# Kill $Process and every process it started (taskkill /T), within a minute whatever happens.
function Stop-Tree([System.Diagnostics.Process]$Process) {
    $kill = New-Object System.Diagnostics.ProcessStartInfo "taskkill.exe"
    $kill.Arguments = "/T /F /PID $($Process.Id)"
    $kill.UseShellExecute = $false
    try {
        $k = [System.Diagnostics.Process]::Start($kill)
        if (-not $k.WaitForExit(30000)) { Write-Host "taskkill /T /PID $($Process.Id) did not finish within 30 s" }
    } catch {
        Write-Host "taskkill failed: $($_.Exception.Message)"
    }
    if (-not $Process.WaitForExit(15000)) {
        try { $Process.Kill() } catch { Write-Host "could not kill pid $($Process.Id): $($_.Exception.Message)" }
        $null = $Process.WaitForExit(15000)
    }
}

# What msiexec's exit code means, for a failure message.
function Get-MsiexecMeaning([int]$Code) {
    switch ($Code) {
        1602 { return "cancelled" }
        1603 { return "fatal error during installation (the log says which action failed)" }
        1618 { return "another installation is in progress (Windows Installer busy)" }
        1619 { return "the package could not be opened" }
        1638 { return "another version of this product is installed" }
        3010 { return "success, a restart held back (a file was in use)" }
        default { return "" }
    }
}

# The last $Lines lines of $Path, if it exists.
function Show-FileTail([string]$Path, [int]$Lines = 80) {
    if ($Path -and (Test-Path -LiteralPath $Path)) {
        Write-Host "---- tail of $Path ----"
        Get-Content -LiteralPath $Path -Tail $Lines -ErrorAction SilentlyContinue | ForEach-Object { Write-Host $_ }
        Write-Host "---- end ----"
    }
}

# The state of the machine for a hang or a failed check: the service, Windows Installer (and
# whether an installation holds its mutex), the processes that matter and the busiest ones, free
# memory, who holds the service's ports, the tails of $Logs and of the service's newest log file,
# and recent Application / System events. Every part is bounded and none can fail the caller.
function Show-Diagnostics {
    param(
        [string[]]$Logs = @(),
        [string]$Service = "BlueOnyxPrismService",
        [int[]]$Ports = @(32168, 32169),
        [string]$DataDir = (Join-Path $env:ProgramData "BlueOnyxPrism")
    )
    $was = $ErrorActionPreference
    $ErrorActionPreference = "Continue"
    try {
        Write-Host "==== diagnostics $(Get-Date -Format o) ===="
        foreach ($svc in @($Service, "msiserver")) {
            $r = Invoke-Bounded -What "sc queryex $svc" -File "sc.exe" -Arguments @("queryex", $svc) -Seconds 20 -Capture -NoDiagnostics
            Write-Host "-- sc queryex $svc --"
            ($r.Output + $r.Errors) | ForEach-Object { Write-Host $_ }
        }
        $mutex = $null
        if ([System.Threading.Mutex]::TryOpenExisting("Global\_MSIExecute", [ref]$mutex)) {
            Write-Host "Windows Installer: an installation is in progress (Global\_MSIExecute exists)."
            $mutex.Dispose()
        } else {
            Write-Host "Windows Installer: no installation in progress."
        }
        # CPU and start time of a protected process cannot be read: -1 / "?" (the CPU property would
        # throw under StrictMode).
        $rows = @(Get-Process -ErrorAction SilentlyContinue | ForEach-Object {
                $cpu = -1
                $start = "?"
                try { $cpu = [int]$_.TotalProcessorTime.TotalSeconds } catch { $cpu = -1 }
                try { $start = $_.StartTime.ToString("HH:mm:ss") } catch { $start = "?" }
                [pscustomobject]@{ Id = $_.Id; Name = $_.ProcessName; "CPU s" = $cpu; "WS MB" = [int]($_.WorkingSet64 / 1MB); Started = $start }
            })
        Write-Host "-- processes that matter --"
        $rows | Where-Object { $_.Name -match '^(msiexec|blue-onyx-prism|sc|icacls|cmd|curl|pwsh|powershell|Runner\.)' } |
            Sort-Object Name, Id | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
        Write-Host "-- busiest processes --"
        $rows | Sort-Object "CPU s" -Descending | Select-Object -First 10 | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
        Write-Host "-- largest processes --"
        $rows | Sort-Object "WS MB" -Descending | Select-Object -First 10 | Format-Table -AutoSize | Out-String -Width 200 | Write-Host
        try {
            $os = Get-CimInstance Win32_OperatingSystem -OperationTimeoutSec 20 -ErrorAction Stop
            Write-Host ("Memory: {0:N0} MB free of {1:N0} MB; commit {2:N0} MB free of {3:N0} MB." -f ($os.FreePhysicalMemory / 1KB), ($os.TotalVisibleMemorySize / 1KB), ($os.FreeVirtualMemory / 1KB), ($os.TotalVirtualMemorySize / 1KB))
        } catch {
            Write-Host "Memory: not read ($($_.Exception.Message))."
        }
        $r = Invoke-Bounded -What "netstat -ano" -File "netstat.exe" -Arguments @("-ano", "-p", "TCP") -Seconds 30 -Capture -NoDiagnostics
        $pattern = ($Ports | ForEach-Object { ":$_\s" }) -join "|"
        Write-Host "-- TCP sockets on ports $($Ports -join ', ') (last column: pid) --"
        $r.Output | Where-Object { $_ -match $pattern } | ForEach-Object { Write-Host $_ }
        foreach ($l in $Logs) { Show-FileTail $l 80 }
        $logDir = Join-Path $DataDir "logs"
        if (Test-Path -LiteralPath $logDir) {
            $newest = Get-ChildItem -LiteralPath $logDir -File -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending | Select-Object -First 1
            if ($newest) { Show-FileTail $newest.FullName 60 }
        }
        Write-Host "-- recent Application events (MsiInstaller, BlueOnyxPrism) --"
        Get-WinEvent -FilterHashtable @{ LogName = "Application"; ProviderName = @("MsiInstaller", "BlueOnyxPrism") } -MaxEvents 20 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "[$($_.ProviderName)] $($_.TimeCreated) $($_.Message)" }
        Write-Host "-- recent Service Control Manager events --"
        Get-WinEvent -FilterHashtable @{ LogName = "System"; ProviderName = "Service Control Manager" } -MaxEvents 10 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "[scm] $($_.TimeCreated) $($_.Message)" }
        Write-Host "==== end of diagnostics ===="
    } catch {
        Write-Host "diagnostics stopped: $($_.Exception.Message)"
    } finally {
        $ErrorActionPreference = $was
    }
}
