<#
.SYNOPSIS
    Builds the per-machine Blue Onyx Prism MSI with WiX Toolset v7.
.PARAMETER Version
    Numeric MSI version, x.y.z or x.y.z.w (strip any "v" prefix and pre-release suffix first).
.PARAMETER BinDir
    Folder with blue-onyx-prism-service.exe, blue-onyx-prism.exe, blue-onyx-prism-tray.exe and
    blue-onyx-prism-benchmark.exe.
.PARAMETER OutFile
    Path of the .msi to write.
.PARAMETER RepoRoot
    Repository root (default: two levels above this script).
#>
param(
    [Parameter(Mandatory)][string]$Version,
    [Parameter(Mandatory)][string]$BinDir,
    [Parameter(Mandatory)][string]$OutFile,
    [string]$RepoRoot = (Split-Path -Parent (Split-Path -Parent $PSScriptRoot))
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$WixVersion = "7.0.0"
$Extensions = "WixToolset.Util.wixext", "WixToolset.Firewall.wixext", "WixToolset.UI.wixext"

if ($Version -notmatch '^\d+\.\d+\.\d+(\.\d+)?$') {
    throw "Version '$Version' must be numeric x.y.z (strip any 'v' prefix and pre-release suffix)."
}
$BinDir = (Resolve-Path $BinDir).Path.TrimEnd('\')
$RepoRoot = (Resolve-Path $RepoRoot).Path.TrimEnd('\')
foreach ($f in "blue-onyx-prism-service.exe", "blue-onyx-prism.exe", "blue-onyx-prism-tray.exe", "blue-onyx-prism-benchmark.exe") {
    if (-not (Test-Path (Join-Path $BinDir $f))) { throw "$f not found in $BinDir" }
}

# The same rule as scripts/ci/retry.sh, which this script cannot call: it also runs outside Git
# Bash, and from PowerShell `bash` can be WSL's stub. These two commands reach nuget.org, where one
# dropped connection would otherwise fail the build.
function Invoke-WithRetry {
    param([int]$Attempts, [int]$DelaySeconds, [string]$Command, [string[]]$Arguments)
    for ($i = 1; $i -le $Attempts; $i++) {
        & $Command @Arguments | Out-Host
        $rc = $LASTEXITCODE
        if ($rc -eq 0) { return }
        if ($i -lt $Attempts) {
            Write-Host "::warning::attempt $i of $Attempts failed (exit $rc): $Command $Arguments; retrying in $DelaySeconds s"
            Start-Sleep -Seconds $DelaySeconds
        }
    }
    throw "$Command $Arguments failed $Attempts times (last exit $rc)"
}

# --- WiX tool + extensions (pinned) -----------------------------------------------------------
# A wix of another version is replaced, not accepted: the pin is what the MSI is built with.
$wixTools = Join-Path $env:USERPROFILE '.dotnet\tools'
if ($env:PATH -notlike "*$wixTools*") { $env:PATH = "$env:PATH;$wixTools" }
$wixHave = $null
if (Get-Command wix -ErrorAction SilentlyContinue) {
    # "7.0.0+<sha>" -> "7.0.0"
    $wixHave = (((& wix --version) | Select-Object -First 1) -split '\+')[0].Trim()
}
if ($wixHave -ne $WixVersion) {
    Write-Host "Installing the wix dotnet tool $WixVersion (found '$wixHave') ..."
    # `update` installs when absent and replaces another version (up or down).
    Invoke-WithRetry -Attempts 4 -DelaySeconds 20 -Command dotnet -Arguments "tool", "update", "--global", "wix", "--version", $WixVersion
    $wixNow = (((& wix --version) | Select-Object -First 1) -split '\+')[0].Trim()
    if ($wixNow -ne $WixVersion) { throw "wix is $wixNow after the update, expected $WixVersion (another wix earlier on PATH?)." }
}
$have = (& wix extension list -g -acceptEula wix7 2>$null) -join "`n"
foreach ($e in $Extensions) {
    if ($have -notmatch [regex]::Escape("$e $WixVersion")) {
        Write-Host "Adding WiX extension $e/$WixVersion ..."
        Invoke-WithRetry -Attempts 4 -DelaySeconds 20 -Command wix -Arguments "extension", "add", "-g", "-acceptEula", "wix7", "$e/$WixVersion"
    }
}

# --- License.rtf from LICENSE (plain text -> RTF, in obj/) --------------------------------------
$obj = Join-Path $PSScriptRoot "obj"
New-Item -ItemType Directory -Force -Path $obj | Out-Null
$rtfPath = Join-Path $obj "License.rtf"
$sb = [System.Text.StringBuilder]::new()
[void]$sb.Append('{\rtf1\ansi\deff0{\fonttbl{\f0\fnil\fcharset0 Consolas;}}\viewkind4\uc1\pard\f0\fs18 ')
foreach ($line in (Get-Content -LiteralPath (Join-Path $RepoRoot "LICENSE"))) {
    $t = $line.Replace('\', '\\').Replace('{', '\{').Replace('}', '\}')
    # Non-ASCII characters become RTF unicode escapes.
    $t = [regex]::Replace($t, '[^\u0000-\u007F]', { param($m) '\u' + [int][char]$m.Value + '?' })
    [void]$sb.Append($t).Append('\par').Append("`r`n")
}
[void]$sb.Append('}')
[System.IO.File]::WriteAllText($rtfPath, $sb.ToString(), [System.Text.Encoding]::ASCII)

# --- Visual C++ runtime, installed next to the exes (stage-vcruntime.ps1, in obj/) ---------------
$vcRuntime = Join-Path $obj "vcruntime"
& (Join-Path $PSScriptRoot "stage-vcruntime.ps1") -Dest $vcRuntime -Cli (Join-Path $BinDir "blue-onyx-prism.exe")

# --- Build -------------------------------------------------------------------------------------
$out = [System.IO.Path]::GetFullPath($OutFile)
New-Item -ItemType Directory -Force -Path (Split-Path -Parent $out) | Out-Null
Write-Host "Building $out (version $Version) ..."
$wixArgs = @(
    "build", (Join-Path $PSScriptRoot "main.wxs"),
    "-acceptEula", "wix7",
    "-arch", "x64",
    "-d", "Version=$Version",
    "-d", "BinDir=$BinDir",
    "-d", "RepoRoot=$RepoRoot",
    "-d", "LicenseRtf=$rtfPath",
    "-d", "VcRuntimeDir=$vcRuntime",
    "-ext", "WixToolset.Util.wixext",
    "-ext", "WixToolset.Firewall.wixext",
    "-ext", "WixToolset.UI.wixext",
    "-o", $out
)
& wix @wixArgs
if ($LASTEXITCODE -ne 0) { throw "wix build failed (exit $LASTEXITCODE)" }
Write-Host "Built $out"
