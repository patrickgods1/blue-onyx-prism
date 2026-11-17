<#
.SYNOPSIS
    Print the machine's state for a Windows installer test that ran out of time
    (packaging/smoke-test-installers.sh runs it when the smoke test passes its limit): the service,
    Windows Installer, processes, memory, ports, the newest msiexec log of -LogDir and the service's
    log. Changes nothing.
.PARAMETER LogDir
    Folder of the msiexec logs (msi-*.log).
#>
param([string]$LogDir = ".")

Set-StrictMode -Version Latest
$ErrorActionPreference = "Continue"
. (Join-Path $PSScriptRoot "bounded.ps1")

$newest = @(Get-ChildItem -Path (Join-Path $LogDir "msi-*.log") -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 | ForEach-Object { $_.FullName })
Show-Diagnostics -Logs $newest
