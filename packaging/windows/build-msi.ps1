<#
.SYNOPSIS
    Builds the per-machine Blue Onyx Prism MSI with WiX Toolset v7.
.PARAMETER Version
    Numeric MSI version, x.y.z or x.y.z.w (strip any "v" prefix and pre-release suffix first).
.PARAMETER BinDir
    Folder with blue-onyx-prism-service.exe, blue-onyx-prism.exe and blue-onyx-prism-benchmark.exe.
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
foreach ($f in "blue-onyx-prism-service.exe", "blue-onyx-prism.exe", "blue-onyx-prism-benchmark.exe") {
    if (-not (Test-Path (Join-Path $BinDir $f))) { throw "$f not found in $BinDir" }
}

# --- WiX tool + extensions (pinned) -----------------------------------------------------------
if (-not (Get-Command wix -ErrorAction SilentlyContinue)) {
    Write-Host "Installing the wix dotnet tool $WixVersion ..."
    & dotnet tool install --global wix --version $WixVersion
    if ($LASTEXITCODE -ne 0) { throw "dotnet tool install wix failed (exit $LASTEXITCODE)" }
    $env:PATH = "$env:PATH;$(Join-Path $env:USERPROFILE '.dotnet\tools')"
}
$have = (& wix extension list -g -acceptEula wix7 2>$null) -join "`n"
foreach ($e in $Extensions) {
    if ($have -notmatch [regex]::Escape("$e $WixVersion")) {
        Write-Host "Adding WiX extension $e/$WixVersion ..."
        & wix extension add -g -acceptEula wix7 "$e/$WixVersion"
        if ($LASTEXITCODE -ne 0) { throw "wix extension add $e failed (exit $LASTEXITCODE)" }
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
    "-ext", "WixToolset.Util.wixext",
    "-ext", "WixToolset.Firewall.wixext",
    "-ext", "WixToolset.UI.wixext",
    "-o", $out
)
& wix @wixArgs
if ($LASTEXITCODE -ne 0) { throw "wix build failed (exit $LASTEXITCODE)" }
Write-Host "Built $out"
