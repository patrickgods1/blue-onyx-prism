<#
.SYNOPSIS
    Copies the Visual C++ runtime DLLs (vcruntime140.dll, vcruntime140_1.dll, msvcp140.dll) from
    this machine's Visual Studio redist folder into -Dest, for an app-local install next to the
    exes (release zip and MSI).
.DESCRIPTION
    The exes and the OpenVINO / ONNX Runtime DLLs they load link the runtime dynamically. A
    Windows without the VC++ redistributable (or with an old one) could not start them; with
    these files next to the exe, Windows takes them first (the application folder comes before
    System32 in the DLL search order), for the downloaded runtimes too.

    The files come from the newest Visual Studio with the C++ tools (vswhere), its default redist
    version (VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt), folder
    VC\Redist\MSVC\<version>\x64\Microsoft.VC*.CRT. Each must be Microsoft-signed and at least
    -MinVersion; the versions and SHA-256 are printed, so a release's log names what it shipped.
    Redistributable under Microsoft's "Distributable Code" terms for Visual Studio (the files in
    VC\Redist), see THIRD_PARTY_LICENSES.md.
.PARAMETER Dest
    Folder to copy the DLLs into (created when missing).
.PARAMETER MinVersion
    Lowest acceptable file version. 14.40 (Visual Studio 2022 17.10) changed std::mutex: code
    built with it crashes on an older msvcp140.dll, and the pinned OpenVINO / ONNX Runtime builds
    are newer than that.
#>
param(
    [Parameter(Mandatory)][string]$Dest,
    [string]$MinVersion = "14.40.33810.0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Files = "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"

$vswhere = Join-Path ${env:ProgramFiles(x86)} "Microsoft Visual Studio\Installer\vswhere.exe"
if (-not (Test-Path $vswhere)) {
    throw "vswhere.exe not found at $vswhere. Install Visual Studio or the Build Tools with the C++ workload (the GitHub windows runners have it)."
}
$vs = & $vswhere -latest -products * -requires Microsoft.VisualStudio.Component.VC.Tools.x86.x64 -property installationPath
if (-not $vs) { throw "vswhere found no Visual Studio with the C++ x64 tools (Microsoft.VisualStudio.Component.VC.Tools.x86.x64)." }
$vs = "$vs".Trim()
$redistRoot = Join-Path $vs "VC\Redist\MSVC"
$versionFile = Join-Path $vs "VC\Auxiliary\Build\Microsoft.VCRedistVersion.default.txt"
if (Test-Path $versionFile) {
    $redistVersion = (Get-Content -LiteralPath $versionFile -Raw).Trim()
} else {
    # No default file: the newest numbered folder.
    $redistVersion = Get-ChildItem -LiteralPath $redistRoot -Directory |
        Where-Object { $_.Name -match '^\d+(\.\d+)+$' } |
        Sort-Object { [version]$_.Name } -Descending | Select-Object -First 1 -ExpandProperty Name
}
if (-not $redistVersion) { throw "No redist version under $redistRoot." }
$x64 = Join-Path $redistRoot "$redistVersion\x64"
$crt = Get-ChildItem -LiteralPath $x64 -Directory -Filter "Microsoft.VC*.CRT" -ErrorAction SilentlyContinue |
    Sort-Object Name -Descending | Select-Object -First 1
if (-not $crt) { throw "No Microsoft.VC*.CRT folder in $x64 (Visual Studio at $vs, redist $redistVersion)." }
Write-Host "Visual C++ runtime from $($crt.FullName) (Visual Studio at $vs, redist $redistVersion)"

New-Item -ItemType Directory -Force -Path $Dest | Out-Null
$min = [version]$MinVersion
foreach ($name in $Files) {
    $src = Join-Path $crt.FullName $name
    if (-not (Test-Path -LiteralPath $src)) { throw "$name is missing from $($crt.FullName)." }
    $info = (Get-Item -LiteralPath $src).VersionInfo
    $version = [version]("{0}.{1}.{2}.{3}" -f $info.FileMajorPart, $info.FileMinorPart, $info.FileBuildPart, $info.FilePrivatePart)
    if ($version -lt $min) {
        throw "$src is version $version, older than $min: code built with Visual Studio 17.10+ (the pinned runtimes) can crash on it. Update Visual Studio on the build machine."
    }
    $sig = Get-AuthenticodeSignature -LiteralPath $src
    if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
        throw "$src is not validly signed by Microsoft (status $($sig.Status), signer '$($sig.SignerCertificate.Subject)')."
    }
    Copy-Item -LiteralPath $src -Destination (Join-Path $Dest $name) -Force
    $hash = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLowerInvariant()
    Write-Host "  $name $version sha256 $hash"
}
