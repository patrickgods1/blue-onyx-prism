#!/usr/bin/env bash
# The two ports of a running service (CI): the web UI answers over HTTPS (certificate checks off:
# it is the service's own self-signed one), it needs a sign-in there, the plain port sends a
# browser to it (308), and Blue Iris's API stays on the plain port without one.
#
# usage: scripts/ci/check_https.sh <http-port> <https-port> [host]
# Used by scripts/ci/smoke.sh, the installer smoke and upgrade tests and the Docker CI. Retries for
# up to a minute while the HTTPS listener comes up (it binds after the plain port).
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "usage: $0 <http-port> <https-port> [host]" >&2
  exit 2
fi
HTTP_PORT=$1
HTTPS_PORT=$2
HOST=${3:-127.0.0.1}
PLAIN="http://$HOST:$HTTP_PORT"
TLS="https://$HOST:$HTTPS_PORT"
fail() { echo "HTTPS CHECK FAILED: $*" >&2; exit 1; }
code() { curl -s --connect-timeout 5 --max-time 20 -o /dev/null -w '%{http_code}' "$@" || true; }

# The sign-in page, or the setup page it sends to before the first account: 200 either way.
got=""
for _ in $(seq 1 30); do
  got=$(code -k -L --max-redirs 3 -H 'Accept: text/html' "$TLS/login")
  [ "$got" = 200 ] && break
  sleep 2
done
[ "$got" = 200 ] || fail "$TLS/login answered '$got', expected 200 (the sign-in or the setup page)"
echo "HTTPS: $TLS/login answers 200"

# The readiness probe is open over TLS too.
got=$(code -k "$TLS/v1/status")
case "$got" in 200 | 503) echo "HTTPS: $TLS/v1/status answers $got" ;; *) fail "$TLS/v1/status answered '$got'" ;; esac

# The Config page needs a sign-in: a browser goes to the login (or setup) page, a script gets 401.
got=$(code -k -H 'Accept: text/html' "$TLS/config")
[ "$got" = 303 ] || fail "$TLS/config without a sign-in answered '$got', expected 303 to the login page"
got=$(code -k "$TLS/config")
[ "$got" = 401 ] || fail "$TLS/config for a script without a token answered '$got', expected 401"

# The plain port sends a browser to HTTPS (the bound HTTPS port, the path kept).
got=$(curl -s --connect-timeout 5 --max-time 20 -H 'Accept: text/html' -o /dev/null \
  -w '%{http_code} %{redirect_url}' "$PLAIN/config" || true)
want="308 $TLS/config"
[ "$got" = "$want" ] || fail "$PLAIN/config answered '$got', expected '$want'"
echo "plain HTTP: $PLAIN/config -> 308 $TLS/config"

# Blue Iris's routes stay on the plain port, with no sign-in.
got=$(code -X POST "$PLAIN/v1/vision/custom/list")
[ "$got" = 200 ] || fail "$PLAIN/v1/vision/custom/list answered '$got', expected 200"
echo "HTTPS check ok (web UI $TLS/, Blue Iris $PLAIN)"
