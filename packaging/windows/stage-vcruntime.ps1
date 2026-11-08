<#
.SYNOPSIS
    Copies the Visual C++ runtime DLLs (vcruntime140.dll, vcruntime140_1.dll, msvcp140.dll) from
    the pinned `tool:vc-runtime` resource into -Dest, for an app-local install next to the exes
    (release zip and MSI).
.DESCRIPTION
    The exes and the OpenVINO / ONNX Runtime DLLs they load link the runtime dynamically. A
    Windows without the VC++ redistributable (or with an old one) could not start them; with
    these files next to the exe, Windows takes them first (the application folder comes before
    System32 in the DLL search order), for the downloaded runtimes too.

    The files do not depend on the build machine: -Cli (the built blue-onyx-prism.exe) runs
    `fetch --resource tool:vc-runtime` against a throwaway config in a temp folder, which downloads
    the pinned msvc-runtime wheel (URL, size, SHA-256 and a SHA-256 per DLL in
    src/resources/catalog/tools.rs) and extracts the DLLs to <temp>\tools\vc-runtime. Each copy
    must be Microsoft-signed, at least -MinVersion and exactly -FileVersion; the versions and
    SHA-256 are printed, so a release's log names what it shipped. Redistributable under
    Microsoft's "Distributable Code" terms for Visual Studio, see THIRD_PARTY_LICENSES.md.
.PARAMETER Dest
    Folder to copy the DLLs into (created when missing).
.PARAMETER Cli
    Path of the built blue-onyx-prism.exe (the CLI, not the service).
.PARAMETER MinVersion
    Lowest acceptable file version. 14.40 (Visual Studio 2022 17.10) changed std::mutex: code
    built with it crashes on an older msvcp140.dll, and the pinned OpenVINO / ONNX Runtime builds
    are newer than that.
.PARAMETER FileVersion
    The exact FileVersion every DLL must have: VC_RUNTIME_FILE_VERSION in
    src/resources/catalog/tools.rs (change both together).
#>
param(
    [Parameter(Mandatory)][string]$Dest,
    [Parameter(Mandatory)][string]$Cli,
    [string]$MinVersion = "14.40.33810.0",
    [string]$FileVersion = "14.44.35211.0"
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"
$Files = "vcruntime140.dll", "vcruntime140_1.dll", "msvcp140.dll"

if (-not (Test-Path -LiteralPath $Cli)) { throw "-Cli $Cli does not exist: build blue-onyx-prism.exe first." }
$Cli = (Resolve-Path -LiteralPath $Cli).Path

# A throwaway data root: the config's download_dir points at it, so nothing is written next to the
# exe or into the repo. `fetch --resource` downloads only the named resource (no config-driven
# downloads), and the CLI saves nothing but this config.
$work = Join-Path ([System.IO.Path]::GetTempPath()) ("bop-vcruntime-" + [guid]::NewGuid().ToString("N"))
New-Item -ItemType Directory -Force -Path $work | Out-Null
try {
    $cfg = Join-Path $work "config.json"
    [System.IO.File]::WriteAllText($cfg, (@{ download_dir = $work } | ConvertTo-Json -Compress), (New-Object System.Text.UTF8Encoding($false)))

    $attempts = 4
    for ($i = 1; $i -le $attempts; $i++) {
        & $Cli --config $cfg fetch --resource tool:vc-runtime | Out-Host
        if ($LASTEXITCODE -eq 0) { break }
        if ($i -eq $attempts) { throw "fetch --resource tool:vc-runtime failed $attempts times (last exit $LASTEXITCODE)" }
        Write-Host "::warning::attempt $i of $attempts failed (exit $LASTEXITCODE); retrying in 20 s"
        Start-Sleep -Seconds 20
    }
    $srcDir = Join-Path $work "tools\vc-runtime"   # VC_RUNTIME_DEST
    Write-Host "Visual C++ runtime from the pinned wheel (tool:vc-runtime), extracted to $srcDir"

    New-Item -ItemType Directory -Force -Path $Dest | Out-Null
    $min = [version]$MinVersion
    $exact = [version]$FileVersion
    foreach ($name in $Files) {
        $src = Join-Path $srcDir $name
        if (-not (Test-Path -LiteralPath $src)) { throw "$name is missing from $srcDir." }
        $info = (Get-Item -LiteralPath $src).VersionInfo
        $version = [version]("{0}.{1}.{2}.{3}" -f $info.FileMajorPart, $info.FileMinorPart, $info.FileBuildPart, $info.FilePrivatePart)
        if ($version -lt $min) {
            throw "$src is version $version, older than ${min}: code built with Visual Studio 17.10+ (the pinned runtimes) can crash on it."
        }
        if ($version -ne $exact) {
            throw "$src is version $version, the pin says ${exact} (VC_RUNTIME_FILE_VERSION in src/resources/catalog/tools.rs). Update the pin and this script's -FileVersion together."
        }
        $sig = Get-AuthenticodeSignature -LiteralPath $src
        if ($sig.Status -ne "Valid" -or $sig.SignerCertificate.Subject -notmatch 'O=Microsoft Corporation') {
            throw "$src is not validly signed by Microsoft (status $($sig.Status), signer '$($sig.SignerCertificate.Subject)')."
        }
        Copy-Item -LiteralPath $src -Destination (Join-Path $Dest $name) -Force
        $hash = (Get-FileHash -LiteralPath $src -Algorithm SHA256).Hash.ToLowerInvariant()
        Write-Host "  $name $version sha256 $hash"
    }
} finally {
    Remove-Item -LiteralPath $work -Recurse -Force -ErrorAction SilentlyContinue
}
