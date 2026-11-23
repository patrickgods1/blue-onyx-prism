# Resource measurement harness for Windows 11 (PowerShell 5.1, built-ins only).
#
# Same plan as measure_resources.sh: start the release service on a scratch config and an empty
# scratch data root, sample it through idle / load / idle / generation restart / idle, then check
# what Stop-Process -Force and a graceful stop leave behind. A report: always exits 0.
# See scripts/research/README_measure.md.
#
# Usage: powershell -ExecutionPolicy Bypass -File measure_resources.ps1 [-BinDir target\release]
#   [-Scratch DIR] [-Model IPcam-general] [-Idle 600] [-LoadRate 10] [-LoadSecs 300]
#   [-PostIdle 300] [-RestartIdle 300] [-Interval 5] [-FinalRun 60]
param(
    [string]$BinDir = 'target\release',
    [string]$Scratch = '',
    [string]$Model = 'IPcam-general',
    [int]$Idle = 600,
    [double]$LoadRate = 10,
    [int]$LoadSecs = 300,
    [int]$PostIdle = 300,
    [int]$RestartIdle = 300,
    [int]$Interval = 5,
    [int]$FinalRun = 60
)
$ErrorActionPreference = 'Continue'

function Need($path, $what) {
    if (-not (Test-Path $path)) {
        Write-Output "missing prerequisite: $path ($what)"
        Write-Output "Build with 'cargo build --release' and 'blue-onyx-prism download-models --name $Model' in $BinDir, then run again."
        exit 0
    }
}
$BinDirAbs = (Resolve-Path $BinDir -ErrorAction SilentlyContinue)
if (-not $BinDirAbs) { Write-Output "missing prerequisite: $BinDir"; exit 0 }
$BinDirAbs = $BinDirAbs.Path
$Bin = Join-Path $BinDirAbs 'blue-onyx-prism.exe'
$LoadTest = Join-Path $BinDirAbs 'test-blue-onyx-prism.exe'
$ModelFile = Join-Path $BinDirAbs "models\$Model.onnx"
Need $Bin 'release binary'
Need $LoadTest 'load test binary'
Need $ModelFile 'model file; only .onnx models are supported by this script'

if (-not $Scratch) { $Scratch = Join-Path ([IO.Path]::GetTempPath()) ('bop-measure-' + [guid]::NewGuid().ToString('N').Substring(0, 8)) }
New-Item -ItemType Directory -Force -Path $Scratch | Out-Null
$Scratch = (Resolve-Path $Scratch).Path
$Data = Join-Path $Scratch 'data'
New-Item -ItemType Directory -Force -Path $Data | Out-Null
$Cfg = Join-Path $Scratch 'config.json'
$Csv = Join-Path $Scratch 'samples.csv'
$Log = Join-Path $Scratch 'service.log'

function PortInUse($p) {
    $c = New-Object Net.Sockets.TcpClient
    try { $c.Connect('127.0.0.1', $p); return $true } catch { return $false } finally { $c.Close() }
}
$Port = Get-Random -Minimum 20000 -Maximum 40000
while ((PortInUse $Port) -or (PortInUse ($Port + 1))) { $Port += 2 }
$HPort = $Port + 1

# Runtimes are reused from the binary dir by path; the data root stays empty.
$cfgObj = [ordered]@{
    format = 1; port = $Port; https_port = $HPort; bind_address = '127.0.0.1'
    download_dir = $Data; auto_download = $false
    openvino_dir = $(if (Test-Path (Join-Path $BinDirAbs 'openvino')) { Join-Path $BinDirAbs 'openvino' } else { $null })
    onnxruntime_dir = $(if (Test-Path (Join-Path $BinDirAbs 'onnxruntime')) { Join-Path $BinDirAbs 'onnxruntime' } else { $null })
    models = @(@{ name = $Model; path = $ModelFile; family = 'yolo5'; enabled = $true })
}
# PowerShell 5.1 writes a BOM with -Encoding UTF8; the service wants plain JSON, so write bytes.
[IO.File]::WriteAllBytes($Cfg, (New-Object Text.UTF8Encoding($false)).GetBytes(($cfgObj | ConvertTo-Json -Depth 5)))

function Snapshot($path) {
    $root = (Resolve-Path $Data).Path
    Get-ChildItem -Path $Data -Recurse -File -Force -ErrorAction SilentlyContinue |
        ForEach-Object { '{0} {1}' -f $_.FullName.Substring($root.Length), $_.Length } |
        Sort-Object | Set-Content -Path $path
}
function NewFiles($before, $after) {
    $b = @{}
    Get-Content $before | ForEach-Object { $b[$_] = $true }
    Get-Content $after | Where-Object { -not $b.ContainsKey($_) }
}
function Descendants($procId) {
    $kids = Get-CimInstance Win32_Process -Filter "ParentProcessId=$procId" -ErrorAction SilentlyContinue
    foreach ($k in $kids) { $k.ProcessId; Descendants $k.ProcessId }
}
function IsAlive($procId) { return [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue) }

$script:T0 = Get-Date
$script:Rows = @()
$script:LastCpu = $null
"time_s,phase,working_set_kb,private_kb,cpu_pct,threads,handles" | Set-Content $Csv
function Sample($procId, $phase) {
    $p = Get-Process -Id $procId -ErrorAction SilentlyContinue
    if (-not $p) { return }
    $now = Get-Date
    $cpuS = $p.TotalProcessorTime.TotalSeconds
    $pct = 0.0
    if ($script:LastCpu) {
        $dt = ($now - $script:LastCpu.t).TotalSeconds
        if ($dt -gt 0) { $pct = 100 * ($cpuS - $script:LastCpu.c) / $dt / [Environment]::ProcessorCount }
    }
    $script:LastCpu = @{ t = $now; c = $cpuS }
    $row = [pscustomobject]@{
        t = [int]($now - $script:T0).TotalSeconds; phase = $phase
        ws = [int]($p.WorkingSet64 / 1KB); priv = [int]($p.PrivateMemorySize64 / 1KB)
        cpu = [math]::Round($pct, 2); threads = $p.Threads.Count; handles = $p.HandleCount
    }
    $script:Rows += $row
    ('{0},{1},{2},{3},{4},{5},{6}' -f $row.t, $row.phase, $row.ws, $row.priv, $row.cpu, $row.threads, $row.handles) | Add-Content $Csv
}
function RunPhase($procId, $name, $secs) {
    Write-Output "[phase] $name ($secs s)"
    $end = (Get-Date).AddSeconds($secs)
    $script:LastCpu = $null
    while ((Get-Date) -lt $end) { Sample $procId $name; Start-Sleep -Seconds $Interval }
}
function WaitReady($procId, $secs) {
    for ($i = 0; $i -lt $secs; $i++) {
        if (-not (IsAlive $procId)) { return $false }
        try {
            $r = Invoke-WebRequest -UseBasicParsing -Uri "http://127.0.0.1:$Port/v1/status" -TimeoutSec 3
            if ($r.StatusCode -eq 200) { return $true }
        } catch { }
        Start-Sleep -Seconds 1
    }
    return $false
}
function StartService {
    # Console process with its own window-less console; output to files.
    return Start-Process -FilePath $Bin -ArgumentList @('--config', "`"$Cfg`"") -PassThru -WindowStyle Hidden `
        -RedirectStandardOutput $Log -RedirectStandardError ($Log + '.err')
}
function PhaseStat($phase, $col, $fn) {
    $v = $script:Rows | Where-Object { $_.phase -eq $phase } | ForEach-Object { $_.$col }
    if (-not $v) { return 'n/a' }
    switch ($fn) {
        'avg' { return [math]::Round(($v | Measure-Object -Average).Average, 2) }
        'max' { return ($v | Measure-Object -Maximum).Maximum }
        'min' { return ($v | Measure-Object -Minimum).Minimum }
        'last' { return @($v)[-1] }
        'first' { return @($v)[0] }
    }
}

Write-Output "scratch: $Scratch  ports: $Port/$HPort  model: $Model"
Snapshot (Join-Path $Scratch 'snap-0-empty.txt')
$svc = StartService
if (-not (WaitReady $svc.Id 180)) {
    Write-Output "service did not answer GET /v1/status 200 within 180 s; see $Log"
    Stop-Process -Id $svc.Id -Force -ErrorAction SilentlyContinue
    exit 0
}
Write-Output "ready: pid $($svc.Id)"
Snapshot (Join-Path $Scratch 'snap-1-started.txt')

RunPhase $svc.Id 'idle' $Idle
$rssBefore = PhaseStat 'idle' 'priv' 'last'

Write-Output "[phase] load ($LoadRate req/s for $LoadSecs s)"
$lt = Start-Process -FilePath $LoadTest -PassThru -WindowStyle Hidden `
    -ArgumentList @('--url', "http://127.0.0.1:$Port", '--rate', "$LoadRate", '--duration', "$LoadSecs", '--https-port', "$HPort", '--config', "`"$Cfg`"") `
    -RedirectStandardOutput (Join-Path $Scratch 'loadtest.txt') -RedirectStandardError (Join-Path $Scratch 'loadtest.err')
$script:LastCpu = $null
while (IsAlive $lt.Id) { Sample $svc.Id 'load'; Start-Sleep -Seconds $Interval }
RunPhase $svc.Id 'post-load-idle' $PostIdle
$rssAfterLoad = PhaseStat 'post-load-idle' 'priv' 'last'

# Generation restart: POST /config/restart needs HTTPS and the API token. The certificate is
# self-signed and the request goes to 127.0.0.1 on this machine only, so skipping validation
# (curl.exe -k, built into Windows 10+) cannot be intercepted off-machine.
Write-Output '[phase] generation restart'
$token = (& $Bin --config $Cfg admin-token 2>$null | Select-Object -First 1)
$restartCode = 'none'
if ($token) {
    $restartCode = (& curl.exe -sk -o NUL -w '%{http_code}' -X POST -H "Authorization: Bearer $token" "https://127.0.0.1:$HPort/config/restart")
}
Write-Output "restart request answered: $restartCode"
Start-Sleep -Seconds 4
$restartOk = WaitReady $svc.Id 120
RunPhase $svc.Id 'after-restart' $RestartIdle
$rssAfterRestart = PhaseStat 'after-restart' 'priv' 'last'
Snapshot (Join-Path $Scratch 'snap-2-after-run.txt')

# Stop-Process -Force = TerminateProcess, the closest thing to kill -9.
$kids = @(Descendants $svc.Id)
Write-Output "[Stop-Process -Force] pid $($svc.Id), children: $(if ($kids) { $kids -join ' ' } else { 'none' })"
Stop-Process -Id $svc.Id -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 3
$survivors = @(@($svc.Id) + $kids | Where-Object { IsAlive $_ })
Snapshot (Join-Path $Scratch 'snap-3-after-kill.txt')
$killLeft = @(NewFiles (Join-Path $Scratch 'snap-1-started.txt') (Join-Path $Scratch 'snap-3-after-kill.txt')).Count
$survivors | ForEach-Object { Stop-Process -Id $_ -Force -ErrorAction SilentlyContinue }

Write-Output "[restart after kill] $FinalRun s"
$svc = StartService
$afterCrashOk = WaitReady $svc.Id 180
if ($afterCrashOk) { Start-Sleep -Seconds $FinalRun }
$kids2 = @(Descendants $svc.Id)

# Graceful stop. A hidden console process cannot receive Ctrl-C from here; CloseMainWindow needs
# a window. The service is a console app, so send Ctrl+C by attaching to its console is not
# possible without P/Invoke. Use the installed Windows service when there is one (sc.exe stop
# BlueOnyxPrism) and otherwise report that only a forced stop was possible.
$graceful = 'not measured'
$svcInstalled = [bool](Get-Service -Name 'BlueOnyxPrism' -ErrorAction SilentlyContinue)
if ($svcInstalled) {
    $graceful = 'an installed BlueOnyxPrism service exists: stop it with `blue-onyx-prism service stop` (admin) and rerun to measure it'
}
$null = $svc.CloseMainWindow()
$t = 0
while ((IsAlive $svc.Id) -and $t -lt 10) { Start-Sleep -Seconds 1; $t++ }
if (IsAlive $svc.Id) {
    $graceful = 'no window to close: needed Stop-Process -Force (a console app cannot be asked to exit from a hidden launch)'
    Stop-Process -Id $svc.Id -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 2
} else { $graceful = "exited $t s after CloseMainWindow" }
Snapshot (Join-Path $Scratch 'snap-4-after-stop.txt')
$stopLeft = @(NewFiles (Join-Path $Scratch 'snap-3-after-kill.txt') (Join-Path $Scratch 'snap-4-after-stop.txt')).Count
$stopSurv = @($kids2 | Where-Object { IsAlive $_ })

function Stable($phase) {
    $mn = PhaseStat $phase 'threads' 'min'; $mx = PhaseStat $phase 'threads' 'max'
    if ($mn -eq 'n/a') { return $true }
    return (($mx - $mn) -le 4)
}
$stable = (Stable 'idle') -and (Stable 'load') -and (Stable 'post-load-idle') -and (Stable 'after-restart')
$h0 = PhaseStat 'idle' 'handles' 'first'; $h1 = PhaseStat 'after-restart' 'handles' 'last'

Write-Output ''
Write-Output '================ summary ================'
Write-Output ("{0,-40} {1}" -f 'scratch dir (samples.csv, logs)', $Scratch)
Write-Output ("{0,-40} {1} %" -f 'idle CPU avg (of all cores)', (PhaseStat 'idle' 'cpu' 'avg'))
Write-Output ("{0,-40} {1} %" -f 'post-load idle CPU avg', (PhaseStat 'post-load-idle' 'cpu' 'avg'))
Write-Output ("{0,-40} {1} %" -f 'after-restart idle CPU avg', (PhaseStat 'after-restart' 'cpu' 'avg'))
Write-Output ("{0,-40} {1}" -f 'idle wakeups', 'not measurable without ETW / admin (use Resource Monitor or xperf)')
Write-Output ("{0,-40} {1} kB private (working set {2} kB)" -f 'before load (end of idle)', $rssBefore, (PhaseStat 'idle' 'ws' 'last'))
Write-Output ("{0,-40} {1} kB private (peak under load {2} kB)" -f 'after load + idle', $rssAfterLoad, (PhaseStat 'load' 'priv' 'max'))
Write-Output ("{0,-40} {1} kB private" -f 'after generation restart + idle', $rssAfterRestart)
Write-Output ("{0,-40} {1}, back up: {2}" -f 'generation restart (HTTP code)', $restartCode, $restartOk)
Write-Output ("{0,-40} {1}" -f 'thread count stable (within 4)', $stable)
Write-Output ("{0,-40} {1} -> {2}" -f 'handle count idle start -> end', $h0, $h1)
Write-Output ("{0,-40} {1}" -f 'children at kill', $(if ($kids) { $kids -join ' ' } else { 'none' }))
Write-Output ("{0,-40} {1}" -f 'survivors after Stop-Process -Force', $(if ($survivors) { $survivors -join ' ' } else { 'none' }))
Write-Output ("{0,-40} {1} files" -f 'new/changed files after kill', $killLeft)
Write-Output ("{0,-40} {1}" -f 'restart after kill served', $afterCrashOk)
Write-Output ("{0,-40} {1}" -f 'graceful stop', $graceful)
Write-Output ("{0,-40} {1}" -f 'survivors after graceful stop', $(if ($stopSurv) { $stopSurv -join ' ' } else { 'none' }))
Write-Output ("{0,-40} {1} files" -f 'new/changed files after graceful stop', $stopLeft)
Write-Output ''
Write-Output 'Files new since the empty start (after final stop):'
NewFiles (Join-Path $Scratch 'snap-0-empty.txt') (Join-Path $Scratch 'snap-4-after-stop.txt') | Select-Object -First 40
exit 0
