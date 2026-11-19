<#
.SYNOPSIS
    Open-loop bursts against a running Blue Onyx Prism service, once per OpenVINO
    `performance_hint`, collected into one table (plan phase 3.6, docs/PRODUCTION_REVIEW.md
    "Pending measurement: OpenVINO GPU performance hint").
.DESCRIPTION
    For each hint in -Hints the script waits until the service runs the model with that hint, then
    for each rate in -Rates runs
        test-blue-onyx-prism --url <Url> [--model <Model>] --rate <r> --duration <Duration>
    and reads its summary: requests answered, 503s (overload), failures, answers per second and the
    latency percentiles from the scheduled start (what a camera trigger waits, client lag included).
    One discarded warm-up run (-WarmupSeconds at the lowest rate) precedes each hint's runs.

    It never changes the service's configuration. Switch the hint yourself when it asks: put
    "performance_hint": "<hint>" in the model's entry of the service's config file (a model's own
    value reloads that model within a few seconds; the global key needs a restart), wait until the
    Config page or the log shows the model Ready again (the log line "compiled on GPU ...
    PERFORMANCE_HINT=..." names the hint and OPTIMAL_NUMBER_OF_INFER_REQUESTS), then press Enter.
    -NoPrompt skips the question (one hint per invocation: pass the one the service runs now,
    -Append adds its rows to the same CSV).

    Read-only on everything but -Out. Uses the GPU only through the service under test.
.PARAMETER Url
    Service base URL (plain HTTP port; default http://127.0.0.1:32168).
.PARAMETER Model
    Model name for POST /v1/vision/custom/{model}; empty = the default model (/v1/vision/detection).
.PARAMETER Hints
    Hints to measure, in order: latency and/or throughput (default both). A list or one
    comma-separated string (what `powershell -File` passes).
.PARAMETER Rates
    Requests per second to schedule (default 2, 4, 6, 8). A list or one comma-separated string.
.PARAMETER Duration
    Seconds per run (default 60).
.PARAMETER WarmupSeconds
    Seconds of the discarded warm-up run per hint (default 10; 0 = none).
.PARAMETER Client
    Path of test-blue-onyx-prism(.exe); default: next to this repository's target\release or
    target\debug build, else on PATH.
.PARAMETER Out
    CSV the rows are written to (default hint_burst-<time>.csv in the current folder).
.PARAMETER Append
    Add to -Out instead of replacing it.
.PARAMETER NoPrompt
    Do not wait for Enter before each hint.
.EXAMPLE
    # Dev box, UHD 630, model yolo26n on openvino:gpu.0; switch the hint when asked.
    powershell -ExecutionPolicy Bypass -File scripts\research\hint_burst.ps1 -Model yolo26n -Rates 2,4,6,8 -Duration 60
.EXAMPLE
    # One hint per call (the service already runs throughput), appended to an earlier CSV.
    .\scripts\research\hint_burst.ps1 -Model yolo26n -Hints throughput -NoPrompt -Append -Out hints.csv
#>
[CmdletBinding()]
param(
    [string]$Url = 'http://127.0.0.1:32168',
    [string]$Model = '',
    [string[]]$Hints = @('latency', 'throughput'),
    [string[]]$Rates = @('2', '4', '6', '8'),
    [ValidateRange(1, 3600)]
    [double]$Duration = 60,
    [ValidateRange(0, 600)]
    [double]$WarmupSeconds = 10,
    [string]$Client = '',
    [string]$Out = '',
    [switch]$Append,
    [switch]$NoPrompt
)

Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
$inv = [Globalization.CultureInfo]::InvariantCulture

# `-File` hands "2,4,8" over as one string: split every item on commas.
function Split-List([string[]]$items) {
    @($items | ForEach-Object { $_ -split ',' } | ForEach-Object { $_.Trim() } | Where-Object { $_ })
}
$HintList = Split-List $Hints | ForEach-Object { $_.ToLowerInvariant() }
foreach ($h in $HintList) {
    if ($h -notin 'latency', 'throughput') { throw "-Hints: '$h' is not latency or throughput" }
}
$RateList = Split-List $Rates | ForEach-Object {
    $r = 0.0
    if (-not [double]::TryParse($_, [Globalization.NumberStyles]::Float, $inv, [ref]$r) -or $r -le 0) {
        throw "-Rates: '$_' is not a positive number of requests per second"
    }
    $r
}
if (-not $HintList -or -not $RateList) { throw 'nothing to measure: -Hints and -Rates need a value each' }

function Find-Client {
    if ($Client) {
        if (-not (Test-Path -LiteralPath $Client -PathType Leaf)) { throw "-Client '$Client' does not exist" }
        return (Resolve-Path -LiteralPath $Client).Path
    }
    $repo = Split-Path -Parent (Split-Path -Parent $PSScriptRoot)
    foreach ($kind in 'release', 'debug') {
        foreach ($name in 'test-blue-onyx-prism.exe', 'test-blue-onyx-prism') {
            $p = Join-Path $repo (Join-Path 'target' (Join-Path $kind $name))
            if (Test-Path -LiteralPath $p -PathType Leaf) { return $p }
        }
    }
    $cmd = Get-Command 'test-blue-onyx-prism' -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw 'test-blue-onyx-prism not found: build it (cargo build --bin test-blue-onyx-prism) or pass -Client'
}

# Number after `label` on a summary line ("p50     12.3" -> 12.3); NaN when absent.
function Get-Field([string]$line, [string]$label) {
    $m = [regex]::Match($line, [regex]::Escape($label) + '\s+(-?[0-9.]+|NaN)')
    if ($m.Success -and $m.Groups[1].Value -ne 'NaN') {
        return [double]::Parse($m.Groups[1].Value, [Globalization.CultureInfo]::InvariantCulture)
    }
    return [double]::NaN
}

function Invoke-Burst([string]$exe, [double]$rate, [double]$seconds) {
    $argv = @('--url', $Url, '--rate', $rate.ToString($inv), '--duration', $seconds.ToString($inv))
    if ($Model) { $argv += @('--model', $Model) }
    # A run with any failed request exits 1; its summary is still printed and still counts.
    $prev = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    try {
        $text = & $exe @argv 2>&1 | ForEach-Object { "$_" }
    } finally {
        $ErrorActionPreference = $prev
    }
    $counts = $text | Where-Object { $_ -match 'answered 503' } | Select-Object -First 1
    $sched = $text | Where-Object { $_ -match '^\s*from scheduled start' } | Select-Object -First 1
    if (-not $counts) {
        Write-Warning ("no summary from the load test:`n" + ($text -join "`n"))
        return $null
    }
    $m = [regex]::Match($counts, '(\d+) ok, (\d+) answered 503 \(overload\), (\d+) failed, (\d+) without an answer; ([0-9.]+) ok/s')
    if (-not $m.Success) { throw "unexpected summary line: $counts" }
    $g = $m.Groups
    $line = if ($sched) { $sched } else { '' }
    [pscustomobject]@{
        ok        = [int]$g[1].Value
        status503 = [int]$g[2].Value
        failed    = [int]$g[3].Value
        no_answer = [int]$g[4].Value
        fps       = [double]::Parse($g[5].Value, $inv)
        p50_ms    = Get-Field $line 'p50'
        p95_ms    = Get-Field $line 'p95'
        p99_ms    = Get-Field $line 'p99'
        max_ms    = Get-Field $line 'max'
    }
}

$exe = Find-Client
if (-not $Out) { $Out = 'hint_burst-{0}.csv' -f (Get-Date -Format 'yyyyMMdd-HHmmss') }
$target = if ($Model) { "model '$Model'" } else { 'the default model' }
Write-Host "load test: $exe"
Write-Host "service:   $Url, $target; rates $($RateList -join ', ') req/s, $Duration s each"

$rows = New-Object System.Collections.Generic.List[object]
foreach ($hint in $HintList) {
    if (-not $NoPrompt) {
        Write-Host ''
        Write-Host "Set `"performance_hint`": `"$hint`" on $target in the service config, wait until it is Ready again"
        [void](Read-Host 'then press Enter')
    }
    if ($WarmupSeconds -gt 0) {
        $low = ($RateList | Measure-Object -Minimum).Minimum
        Write-Host "[$hint] warm-up: $low req/s for $WarmupSeconds s (discarded)"
        [void](Invoke-Burst $exe $low $WarmupSeconds)
    }
    foreach ($rate in $RateList) {
        Write-Host "[$hint] $rate req/s for $Duration s"
        $r = Invoke-Burst $exe $rate $Duration
        if ($null -eq $r) { continue }
        $row = [pscustomobject]@{
            time      = (Get-Date -Format 's')
            hint      = $hint
            model     = $(if ($Model) { $Model } else { '(default)' })
            rate      = $rate
            duration  = $Duration
            fps       = $r.fps
            p50_ms    = $r.p50_ms
            p95_ms    = $r.p95_ms
            p99_ms    = $r.p99_ms
            max_ms    = $r.max_ms
            ok        = $r.ok
            status503 = $r.status503
            failed    = $r.failed
            no_answer = $r.no_answer
        }
        $rows.Add($row)
        $row | Export-Csv -LiteralPath $Out -NoTypeInformation -Append:($Append -or $rows.Count -gt 1)
    }
}

Write-Host ''
$rows | Format-Table hint, rate, fps, p50_ms, p95_ms, p99_ms, max_ms, ok, status503, failed, no_answer -AutoSize | Out-String | Write-Host
Write-Host "rows written to $Out"
Write-Host 'latency is from the scheduled start; compare p99 with Blue Iris''s timeout before changing a default.'
