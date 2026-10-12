#!/usr/bin/env bash
# Start the server, wait for it, run the HTTP and client checks, assert the execution provider,
# stop the server.
#
# usage: scripts/ci/smoke.sh <server-bin> <client-bin> <port> <expected-provider-substring> [server args...]
#
# The server args are passed to the server after `--config <tmp>/smoke_config_<port>.json --port
# <port>` (model, family, device, ...). Env: MODEL_NAME (default ipcam-bird) is the name the
# model is served under, IMG the test image.
set -euo pipefail

if [ "$#" -lt 4 ]; then
  echo "usage: $0 <server-bin> <client-bin> <port> <expected-provider-substring> [server args...]" >&2
  exit 2
fi
BIN=$1
CLIENT=$2
PORT=$3
EXPECTED=$4
shift 4

MODEL_NAME=${MODEL_NAME:-ipcam-bird}
IMG=${IMG:-tests/data/dog_bike_car.jpg}
TMP=${RUNNER_TEMP:-${TMPDIR:-/tmp}}
LOG="$TMP/blue-onyx-server-$PORT.log"
CONFIG="$TMP/smoke_config_$PORT.json"
URL=http://127.0.0.1:$PORT
rm -f "$CONFIG"

"$BIN" --config "$CONFIG" --port "$PORT" "$@" >"$LOG" 2>&1 &
PID=$!
cleanup() {
  kill "$PID" 2>/dev/null || true
  if [ "${RUNNER_OS:-}" = "Windows" ]; then
    taskkill //F //IM "$(basename "$BIN")" >/dev/null 2>&1 || true
  fi
  echo "---- server log ($PORT) ----"
  tail -n 50 "$LOG" || true
}
trap cleanup EXIT

for _ in $(seq 1 60); do
  curl -sf "$URL/" >/dev/null && break
  sleep 2
done

OUT=""
for _ in $(seq 1 60); do
  OUT=$(curl -s -F image=@"$IMG" -F min_confidence=0.2 "$URL/v1/vision/detection" || true)
  echo "$OUT" | grep -q '"success":true' && break
  sleep 2
done
echo "detection: $OUT"
echo "$OUT" | grep -q '"success":true'
if ! echo "$OUT" | grep -Eq "\"executionProvider\":\"[^\"]*$EXPECTED"; then
  echo "executionProvider does not contain '$EXPECTED'" >&2
  exit 1
fi

LIST=$(curl -s -X POST "$URL/v1/vision/custom/list")
echo "list: $LIST"
echo "$LIST" | grep -q "$MODEL_NAME"

OUT=$(curl -s -F image=@"$IMG" -F min_confidence=0.2 "$URL/v1/vision/custom/$MODEL_NAME")
echo "custom: $OUT"
echo "$OUT" | grep -q '"success":true'

# The client takes its port from the same URL.
"$CLIENT" --url "$URL" --repeat 5 --parallel 2
"$CLIENT" --url "$URL" --model "$MODEL_NAME" --repeat 3
"$CLIENT" --url "$URL" --list
echo "smoke ok ($EXPECTED on port $PORT)"
