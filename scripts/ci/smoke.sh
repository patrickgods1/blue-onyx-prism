#!/usr/bin/env bash
# Start the server, wait for it, run the HTTP and client checks, assert the execution provider,
# stop the server.
#
# usage: scripts/ci/smoke.sh <server-bin> <client-bin> <port> <expected-provider-substring> [server args...]
#
# The server args are passed to the server after `--config <tmp>/smoke_config_<port>.json --port
# <port>` (model, family, device, ...). Env: MODEL_NAME (default ipcam-bird) is the name the
# model is served under, IMG the test image, MIN_COUNT (default 1) the fewest predictions the
# detection must return at min_confidence 0.2, EXPECT_LABEL (optional) a label that must be among
# them. A provider that answers success with no boxes (numeric failure, broken preprocessing)
# fails the test.
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
MIN_COUNT=${MIN_COUNT:-1}
EXPECT_LABEL=${EXPECT_LABEL:-}
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

# Fail at once (with the log, from the trap) when the server process is gone.
alive() {
  if ! kill -0 "$PID" 2>/dev/null; then
    echo "the server exited early (see the log below)" >&2
    exit 1
  fi
}

for _ in $(seq 1 60); do
  alive
  curl -sf "$URL/" >/dev/null && break
  sleep 2
done

OUT=""
for _ in $(seq 1 60); do
  alive
  OUT=$(curl -s -F image=@"$IMG" -F min_confidence=0.2 "$URL/v1/vision/detection" || true)
  echo "$OUT" | grep -q '"success":true' && break
  sleep 2
done
echo "detection: $OUT"
echo "$OUT" | grep -q '"success":true'
# Field values extracted with fixed patterns, then compared literally (EXPECTED may hold regex
# metacharacters such as "(").
PROVIDER=$(printf '%s' "$OUT" | sed -n 's/.*"executionProvider":"\([^"]*\)".*/\1/p')
if ! printf '%s' "$PROVIDER" | grep -qF -- "$EXPECTED"; then
  echo "executionProvider '$PROVIDER' does not contain '$EXPECTED'" >&2
  exit 1
fi
COUNT=$(printf '%s' "$OUT" | sed -n 's/.*"count":\([0-9][0-9]*\).*/\1/p')
if [ -z "$COUNT" ] || [ "$COUNT" -lt "$MIN_COUNT" ]; then
  echo "detection returned ${COUNT:-no} predictions, expected at least $MIN_COUNT" >&2
  exit 1
fi
if [ -n "$EXPECT_LABEL" ] && ! printf '%s' "$OUT" | grep -qF -- "\"label\":\"$EXPECT_LABEL\""; then
  echo "no '$EXPECT_LABEL' among the predictions" >&2
  exit 1
fi

LIST=$(curl -s -X POST "$URL/v1/vision/custom/list")
echo "list: $LIST"
echo "$LIST" | grep -qiF -- "$MODEL_NAME"

OUT=$(curl -s -F image=@"$IMG" -F min_confidence=0.2 "$URL/v1/vision/custom/$MODEL_NAME")
echo "custom: $OUT"
echo "$OUT" | grep -q '"success":true'
alive

# The client takes its port from the same URL.
"$CLIENT" --url "$URL" --repeat 5 --parallel 2
"$CLIENT" --url "$URL" --model "$MODEL_NAME" --repeat 3
"$CLIENT" --url "$URL" --list
echo "smoke ok ($EXPECTED on port $PORT)"
