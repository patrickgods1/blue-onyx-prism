#!/usr/bin/env bash
# Resource measurement harness (macOS + Linux, bash 3.2 compatible, no root).
#
# Starts the release service on a scratch config and an EMPTY scratch data root, samples its
# resource use through idle, load, idle, a generation restart and idle again, then checks what
# kill -9 and a graceful stop leave behind (child processes, files). It is a report: it exits 0
# whatever it finds. See scripts/research/README_measure.md.
#
# Usage: measure_resources.sh [--bin-dir DIR] [--scratch DIR] [--model NAME]
#          [--idle SECS] [--load-rate R] [--load-secs SECS] [--post-idle SECS]
#          [--restart-idle SECS] [--interval SECS] [--final-run SECS]
set -euo pipefail

BIN_DIR="target/release"
SCRATCH=""
MODEL="IPcam-general"
IDLE=600
LOAD_RATE=10
LOAD_SECS=300
POST_IDLE=300
RESTART_IDLE=300
INTERVAL=5
FINAL_RUN=60

while [ $# -gt 0 ]; do
    case "$1" in
        --bin-dir) BIN_DIR="$2"; shift 2 ;;
        --scratch) SCRATCH="$2"; shift 2 ;;
        --model) MODEL="$2"; shift 2 ;;
        --idle) IDLE="$2"; shift 2 ;;
        --load-rate) LOAD_RATE="$2"; shift 2 ;;
        --load-secs) LOAD_SECS="$2"; shift 2 ;;
        --post-idle) POST_IDLE="$2"; shift 2 ;;
        --restart-idle) RESTART_IDLE="$2"; shift 2 ;;
        --interval) INTERVAL="$2"; shift 2 ;;
        --final-run) FINAL_RUN="$2"; shift 2 ;;
        -h|--help) sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown argument: $1 (see --help)"; exit 0 ;;
    esac
done

OS="$(uname -s)"
BIN="$BIN_DIR/blue-onyx-prism"
LOADTEST="$BIN_DIR/test-blue-onyx-prism"
MODEL_FILE="$BIN_DIR/models/$MODEL.onnx"

need() {
    if [ ! -e "$1" ]; then
        echo "missing prerequisite: $1 ($2)"
        echo "Build with 'cargo build --release' and download the model with"
        echo "'blue-onyx-prism download-models --name $MODEL' in $BIN_DIR, then run again."
        exit 0
    fi
}
need "$BIN" "release binary"
need "$LOADTEST" "load test binary (cargo build --release builds it)"
need "$MODEL_FILE" "model file; only .onnx models are supported by this script"

# Absolute paths: the service runs from the scratch dir.
abspath() { (cd "$(dirname "$1")" && printf '%s/%s\n' "$(pwd)" "$(basename "$1")"); }
BIN="$(abspath "$BIN")"
LOADTEST="$(abspath "$LOADTEST")"
MODEL_FILE="$(abspath "$MODEL_FILE")"
BIN_ABS_DIR="$(dirname "$BIN")"

if [ -z "$SCRATCH" ]; then
    SCRATCH="$(mktemp -d "${TMPDIR:-/tmp}/bop-measure.XXXXXX")"
else
    mkdir -p "$SCRATCH"
fi
SCRATCH="$(cd "$SCRATCH" && pwd)"
DATA="$SCRATCH/data"
mkdir -p "$DATA"
CFG="$SCRATCH/config.json"
CSV="$SCRATCH/samples.csv"
LOG="$SCRATCH/service.log"

# A port pair nothing listens on (bash's /dev/tcp probe: success = something answers).
in_use() { (exec 3<>"/dev/tcp/127.0.0.1/$1") 2>/dev/null; }
pick_ports() {
    local p=$((20000 + RANDOM % 20000))
    while in_use "$p" || in_use $((p + 1)); do p=$((p + 2)); done
    PORT=$p
    HPORT=$((p + 1))
}
pick_ports

# Runtimes are reused from the binary dir by path; the data root stays empty.
OV=null
ORT=null
[ -d "$BIN_ABS_DIR/openvino" ] && OV="\"$BIN_ABS_DIR/openvino\""
[ -d "$BIN_ABS_DIR/onnxruntime" ] && ORT="\"$BIN_ABS_DIR/onnxruntime\""
cat > "$CFG" <<EOF
{
  "format": 1,
  "port": $PORT,
  "https_port": $HPORT,
  "bind_address": "127.0.0.1",
  "download_dir": "$DATA",
  "auto_download": false,
  "openvino_dir": $OV,
  "onnxruntime_dir": $ORT,
  "models": [
    { "name": "$MODEL", "path": "$MODEL_FILE", "family": "yolo5", "enabled": true }
  ]
}
EOF

snapshot() {
    (cd "$DATA" && find . -type f -print | LC_ALL=C sort | while IFS= read -r f; do
        printf '%s %s\n' "$f" "$(wc -c < "$f" | tr -d ' ')"
    done)
}
new_files() { # new_files BEFORE AFTER: lines in AFTER that are not in BEFORE (new or changed size)
    LC_ALL=C comm -13 "$1" "$2"
}

descendants() { # all descendant pids of $1 (recursive pgrep -P)
    local kids k
    kids="$(pgrep -P "$1" 2>/dev/null || true)"
    for k in $kids; do
        echo "$k"
        descendants "$k"
    done
}

alive() { kill -0 "$1" 2>/dev/null; }

# Threads, open fds, voluntary context switches (wakeup proxy) of one process.
threads_of() {
    if [ "$OS" = Linux ]; then
        awk '/^Threads:/ {print $2}' "/proc/$1/status" 2>/dev/null || echo 0
    else
        ps -M -p "$1" 2>/dev/null | awk 'NR>1' | wc -l | tr -d ' '
    fi
}
fds_of() {
    if [ "$OS" = Linux ]; then
        ls "/proc/$1/fd" 2>/dev/null | wc -l | tr -d ' '
    else
        lsof -p "$1" 2>/dev/null | awk 'NR>1' | wc -l | tr -d ' '
    fi
}
vctx_of() { # Linux: sum over all threads. macOS: not available without root, 0.
    if [ "$OS" = Linux ]; then
        cat /proc/"$1"/task/*/status 2>/dev/null |
            awk '/^voluntary_ctxt_switches:/ {s += $2} END {print s + 0}'
    else
        echo 0
    fi
}

T0="$(date +%s)"
echo "time_s,phase,rss_kb,cpu_pct,threads,fds,voluntary_ctxt" > "$CSV"
sample() { # sample PID PHASE
    local pid="$1" phase="$2" rss cpu
    alive "$pid" || return 0
    rss="$(ps -o rss= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    cpu="$(ps -o %cpu= -p "$pid" 2>/dev/null | tr -d ' ' || true)"
    echo "$(($(date +%s) - T0)),$phase,${rss:-0},${cpu:-0},$(threads_of "$pid"),$(fds_of "$pid"),$(vctx_of "$pid")" >> "$CSV"
}

# ps %cpu on macOS and Linux is a lifetime average, which hides bursts. Use the delta of
# cumulative CPU time between samples instead for the summary: cputime in seconds.
cpu_secs_of() {
    local t
    t="$(ps -o time= -p "$1" 2>/dev/null | tr -d ' ' || true)"
    # [[dd-]hh:]mm:ss[.ff] -> seconds
    echo "$t" | awk -F'[-:]' '{
        n = NF; s = $n + 0
        if (n >= 2) s += $(n-1) * 60
        if (n >= 3) s += $(n-2) * 3600
        if (n >= 4) s += $(n-3) * 86400
        print s }'
}

wait_ready() { # wait_ready PID SECS
    local pid="$1" n="$2" i=0 code
    while [ "$i" -lt "$n" ]; do
        alive "$pid" || return 1
        code="$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/status" || true)"
        [ "$code" = 200 ] && return 0
        sleep 1
        i=$((i + 1))
    done
    return 1
}

PID=""
start_service() {
    "$BIN" --config "$CFG" < /dev/null >> "$LOG" 2>&1 &
    PID=$!
}

phase() { # phase NAME SECS: sample every INTERVAL for SECS
    local name="$1" secs="$2" end
    end=$(($(date +%s) + secs))
    echo "[phase] $name ($secs s)"
    while [ "$(date +%s)" -lt "$end" ]; do
        sample "$PID" "$name"
        sleep "$INTERVAL"
    done
}

CPU_MARKS="$SCRATCH/cpu-marks.txt" # phase start/end cumulative CPU seconds + wall seconds
: > "$CPU_MARKS"
mark() { echo "$1 $(date +%s) $(cpu_secs_of "$PID")" >> "$CPU_MARKS"; }

snapshot > "$SCRATCH/snap-0-empty.txt"

echo "scratch: $SCRATCH  ports: $PORT/$HPORT  model: $MODEL  os: $OS"
start_service
if ! wait_ready "$PID" 180; then
    echo "service did not answer GET /v1/status 200 within 180 s; see $LOG"
    tail -n 20 "$LOG" || true
    kill -9 "$PID" 2>/dev/null || true
    exit 0
fi
echo "ready: pid $PID"
snapshot > "$SCRATCH/snap-1-started.txt"

mark idle-start
phase idle "$IDLE"
mark idle-end

RSS_BEFORE_LOAD="$(tail -n 1 "$CSV" | cut -d, -f3)"
mark load-start
echo "[phase] load ($LOAD_RATE req/s for $LOAD_SECS s)"
"$LOADTEST" --url "http://127.0.0.1:$PORT" --rate "$LOAD_RATE" --duration "$LOAD_SECS" \
    --https-port "$HPORT" --config "$CFG" > "$SCRATCH/loadtest.txt" 2>&1 &
LT=$!
while alive "$LT"; do
    sample "$PID" load
    sleep "$INTERVAL"
done
wait "$LT" 2>/dev/null || true
mark load-end

mark post-idle-start
phase post-load-idle "$POST_IDLE"
mark post-idle-end
RSS_AFTER_LOAD="$(tail -n 1 "$CSV" | cut -d, -f3)"

# Generation restart: POST /config/restart is admin-only, HTTPS, with the API token. The
# certificate is self-signed and the request goes to 127.0.0.1 on this machine only, so
# `curl -k` (no verification) cannot be intercepted off-machine; the token comes from the CLI.
echo "[phase] generation restart"
TOKEN="$("$BIN" --config "$CFG" admin-token 2>/dev/null || true)"
RESTART_CODE=none
if [ -n "$TOKEN" ]; then
    RESTART_CODE="$(curl -sk -o /dev/null -w '%{http_code}' -X POST \
        -H "Authorization: Bearer $TOKEN" "https://127.0.0.1:$HPORT/config/restart" || true)"
fi
echo "restart request answered: $RESTART_CODE"
sleep 2
RESTART_OK=no
i=0
while [ "$i" -lt 120 ]; do
    if [ "$(curl -s -o /dev/null -w '%{http_code}' "http://127.0.0.1:$PORT/v1/status" || true)" = 200 ] \
        && [ "$i" -gt 2 ]; then
        RESTART_OK=yes
        break
    fi
    sleep 1
    i=$((i + 1))
done
mark restart-idle-start
phase after-restart "$RESTART_IDLE"
mark restart-idle-end
RSS_AFTER_RESTART="$(tail -n 1 "$CSV" | cut -d, -f3)"
snapshot > "$SCRATCH/snap-2-after-run.txt"

# kill -9: record every descendant first, then see which survive.
KIDS="$(descendants "$PID" | tr '\n' ' ')"
echo "[kill -9] pid $PID, children: ${KIDS:-none}"
kill -9 "$PID" 2>/dev/null || true
wait "$PID" 2>/dev/null || true
sleep 3
SURVIVORS=""
for k in $PID $KIDS; do
    if alive "$k"; then SURVIVORS="$SURVIVORS $k"; fi
done
# Anything still listening on our ports or started from the scratch config.
HOLDERS="$(pgrep -f -- "--config $CFG" 2>/dev/null | tr '\n' ' ' || true)"
snapshot > "$SCRATCH/snap-3-after-kill9.txt"
KILL_LEFT="$(new_files "$SCRATCH/snap-1-started.txt" "$SCRATCH/snap-3-after-kill9.txt" | wc -l | tr -d ' ')"
KILL_LEFT_EMPTY="$(new_files "$SCRATCH/snap-0-empty.txt" "$SCRATCH/snap-3-after-kill9.txt" | wc -l | tr -d ' ')"
for k in $SURVIVORS; do kill -9 "$k" 2>/dev/null || true; done

# Second start after the crash, then a graceful stop (SIGTERM).
echo "[restart after kill -9] $FINAL_RUN s"
start_service
if wait_ready "$PID" 180; then
    sleep "$FINAL_RUN"
    AFTER_CRASH_OK=yes
else
    AFTER_CRASH_OK=no
fi
KIDS2="$(descendants "$PID" | tr '\n' ' ')"
echo "[SIGTERM] pid $PID"
kill -TERM "$PID" 2>/dev/null || true
GRACEFUL_SECS=-1
i=0
while [ "$i" -lt 60 ]; do
    if ! alive "$PID"; then GRACEFUL_SECS=$i; break; fi
    sleep 1
    i=$((i + 1))
done
STOP_SURVIVORS=""
for k in $PID $KIDS2; do
    if alive "$k"; then STOP_SURVIVORS="$STOP_SURVIVORS $k"; fi
done
for k in $STOP_SURVIVORS; do kill -9 "$k" 2>/dev/null || true; done
wait "$PID" 2>/dev/null || true
snapshot > "$SCRATCH/snap-4-after-stop.txt"
STOP_LEFT="$(new_files "$SCRATCH/snap-3-after-kill9.txt" "$SCRATCH/snap-4-after-stop.txt" | wc -l | tr -d ' ')"
STOP_LEFT_EMPTY="$(new_files "$SCRATCH/snap-0-empty.txt" "$SCRATCH/snap-4-after-stop.txt" | wc -l | tr -d ' ')"

# ---- summary ----
phase_cpu() { # phase_cpu START_MARK END_MARK: CPU% over the phase from cumulative CPU time
    awk -v a="$1" -v b="$2" '
        $1 == a { t0 = $2; c0 = $3 }
        $1 == b { t1 = $2; c1 = $3 }
        END { if (t1 > t0) printf "%.2f", 100 * (c1 - c0) / (t1 - t0); else printf "n/a" }' "$CPU_MARKS"
}
phase_stat() { # phase_stat PHASE COLUMN FUNC(min|max|avg|first|last)
    awk -F, -v p="$1" -v c="$2" -v f="$3" '
        $2 == p { v = $c + 0; n++; s += v
                  if (n == 1 || v < mn) mn = v
                  if (n == 1 || v > mx) mx = v
                  if (n == 1) first = v
                  last = v }
        END { if (n == 0) { print "n/a"; exit }
              if (f == "min") print mn; else if (f == "max") print mx
              else if (f == "avg") printf "%.0f\n", s / n
              else if (f == "first") print first; else print last }' "$CSV"
}
vctx_rate() { # voluntary context switches per second during a phase (Linux only)
    if [ "$OS" != Linux ]; then echo "n/a (macOS: needs root)"; return; fi
    awk -F, -v p="$1" '
        $2 == p { if (!n) { t0 = $1; v0 = $7 } t1 = $1; v1 = $7; n++ }
        END { if (n > 1 && t1 > t0) printf "%.1f/s", (v1 - v0) / (t1 - t0); else print "n/a" }' "$CSV"
}
THREADS_STABLE=yes
for ph in idle load post-load-idle after-restart; do
    mn="$(phase_stat "$ph" 5 min)"
    mx="$(phase_stat "$ph" 5 max)"
    if [ "$mn" != "n/a" ] && [ "$mx" != "n/a" ] && [ "$((mx - mn))" -gt 4 ]; then
        THREADS_STABLE="no (phase $ph ranged $mn..$mx)"
    fi
done
FD_A="$(phase_stat idle 6 first)"
FD_B="$(phase_stat after-restart 6 last)"
FD_GROWTH="n/a"
case "$FD_A$FD_B" in *[!0-9]*) ;; *) FD_GROWTH=$((FD_B - FD_A)) ;; esac

echo
echo "================ summary ================"
printf '%-38s %s\n' "scratch dir (samples.csv, logs)" "$SCRATCH"
printf '%-38s %s %%\n' "idle CPU (CPU-time delta)" "$(phase_cpu idle-start idle-end)"
printf '%-38s %s %%\n' "post-load idle CPU" "$(phase_cpu post-idle-start post-idle-end)"
printf '%-38s %s %%\n' "after-restart idle CPU" "$(phase_cpu restart-idle-start restart-idle-end)"
printf '%-38s %s\n' "idle wakeups (voluntary ctx switches)" "$(vctx_rate idle)"
printf '%-38s %s kB\n' "RSS before load (end of idle)" "$RSS_BEFORE_LOAD"
printf '%-38s %s kB (peak under load %s kB)\n' "RSS after load + idle" "$RSS_AFTER_LOAD" "$(phase_stat load 3 max)"
printf '%-38s %s kB\n' "RSS after generation restart + idle" "$RSS_AFTER_RESTART"
printf '%-38s %s\n' "generation restart (HTTP code, back up)" "$RESTART_CODE, $RESTART_OK"
printf '%-38s %s\n' "thread count stable (within 4)" "$THREADS_STABLE"
printf '%-38s %s\n' "open fds growth idle-start -> end" "$FD_GROWTH"
printf '%-38s %s\n' "children at kill -9" "${KIDS:-none}"
printf '%-38s %s\n' "survivors after kill -9" "${SURVIVORS:-none}${HOLDERS:+ (still matching config: $HOLDERS)}"
printf '%-38s %s files\n' "new/changed files after kill -9 (vs run)" "$KILL_LEFT"
printf '%-38s %s files\n' "files in data root vs empty start" "$KILL_LEFT_EMPTY"
printf '%-38s %s\n' "restart after crash served" "$AFTER_CRASH_OK"
printf '%-38s %s s (-1 = needed kill -9)\n' "SIGTERM stop took" "$GRACEFUL_SECS"
printf '%-38s %s\n' "survivors after SIGTERM" "${STOP_SURVIVORS:-none}"
printf '%-38s %s files\n' "new/changed after graceful stop" "$STOP_LEFT"
printf '%-38s %s files\n' "files in data root vs empty start" "$STOP_LEFT_EMPTY"
echo "load test: $(tail -n 3 "$SCRATCH/loadtest.txt" | tr '\n' ' ')"
echo
echo "Files new since the empty start (after graceful stop):"
new_files "$SCRATCH/snap-0-empty.txt" "$SCRATCH/snap-4-after-stop.txt" | head -n 40
exit 0
