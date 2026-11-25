# shellcheck shell=sh
# Helpers shared by every Unix install flow: sourced by scripts/systemd_common.sh (Linux archive
# install and the .deb / .rpm scriptlets) and scripts/launchd_common.sh (macOS zip install and the
# .pkg postinstall), never executed. Strictly POSIX sh (Debian's /bin/sh is dash): no `local`, so
# every variable is prefixed `_bop_`. It sets no shell options; the caller runs `set -eu`.
#
# bop_print_urls reads three settings the sourcing file sets: BOP_URL_USER (the service account
# in the `sudo -u` line), BOP_URL_WEB_HOST (the web UI's host) and BOP_URL_IRIS_HOST (the
# placeholder for Blue Iris's address).

# bop_merge_keys KEYS FILE: FILE (pretty JSON as the service writes it) with the member lines KEYS
# after the opening brace. The file's own top-level lines (2-space indent) of those keys are
# dropped first, so no key repeats (Config::load refuses duplicates), and a comma left before the
# closing brace is removed.
bop_merge_keys() {
  KEYS="$1" awk '
    BEGIN {
      n = split(ENVIRON["KEYS"], L, "\n")
      for (i = 1; i <= n; i++) {
        if (L[i] ~ /^  "[^"]*"[ \t]*:/) { k = L[i]; sub(/^  "/, "", k); sub(/".*/, "", k); drop[k] = 1 }
      }
      m = 0
    }
    NR == 1 { out[++m] = $0; out[++m] = ENVIRON["KEYS"]; next }
    {
      if ($0 ~ /^  "[^"]*"[ \t]*:/) { k = $0; sub(/^  "/, "", k); sub(/".*/, "", k); if (k in drop) next }
      out[++m] = $0
    }
    END {
      for (i = 2; i <= m; i++) if (out[i] ~ /^}/ && out[i - 1] ~ /,[ \t]*$/) sub(/,[ \t]*$/, "", out[i - 1])
      for (i = 1; i <= m; i++) print out[i]
    }' "$2"
}

# bop_migrate_config SRC DEST KEYS DATA_DIR: write the service config DEST from SRC, the config of
# an install-dir (by hand) run, with the member lines KEYS (each ending in a comma: download_dir,
# cache_dir, ...) that point the service at its data root DATA_DIR. The caller checks that DEST
# does not exist yet (this never looks). SRC empty or missing: a fresh config of KEYS plus
# models_dir. SRC that sets download_dir to a path (a string, not the null a program-written
# config holds): copied as is, with a note to check it. Pretty JSON (first line `{`): KEYS merged
# in by bop_merge_keys. Anything else: copied, and KEYS printed to stderr for the user to add.
bop_migrate_config() {
  _bop_src="$1"; _bop_dest="$2"; _bop_keys="$3"; _bop_data="$4"
  [ -n "$_bop_src" ] && [ -f "$_bop_src" ] || _bop_src=""
  case "$_bop_keys" in
    *'"log_path"'*) _bop_added="download_dir, cache_dir, log_path" ;;
    *) _bop_added="download_dir, cache_dir" ;;
  esac
  if [ -n "$_bop_src" ] && grep -Eq '"download_dir"[[:space:]]*:[[:space:]]*"' "$_bop_src"; then
    cp "$_bop_src" "$_bop_dest"
    echo "Copied $_bop_src to $_bop_dest; it already sets download_dir: check that it points at $_bop_data."
  elif [ -n "$_bop_src" ] && [ "$(head -n 1 "$_bop_src")" = "{" ]; then
    # Pretty JSON as the service writes it: insert the keys after the opening brace.
    bop_merge_keys "$_bop_keys" "$_bop_src" > "$_bop_dest"
    echo "Moved the config to $_bop_dest ($_bop_added and runtime dirs added)."
  elif [ -n "$_bop_src" ]; then
    cp "$_bop_src" "$_bop_dest"
    echo "Copied $_bop_src to $_bop_dest. Add these keys to it by hand:" >&2
    echo "$_bop_keys" >&2
  else
    printf '{\n%s\n  "models_dir": "models"\n}\n' "$_bop_keys" > "$_bop_dest"
  fi
}

# bop_print_urls CONFIG EXE: the addresses an install prints. The web UI is HTTPS (self-signed
# certificate, a sign-in; the first visit creates the first account with the one-time code that
# `setup-code` prints); Blue Iris uses the plain HTTP port with no account. The ports are the
# config's when it names them (a pretty config, one key per line), else the defaults. The account
# and the host words come from BOP_URL_USER, BOP_URL_WEB_HOST and BOP_URL_IRIS_HOST.
bop_print_urls() {
  _bop_port=$(sed -n 's/^[[:space:]]*"port":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$1" 2>/dev/null | head -n 1 || true)
  _bop_https=$(sed -n 's/^[[:space:]]*"https_port":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$1" 2>/dev/null | head -n 1 || true)
  echo "Web UI:    https://${BOP_URL_WEB_HOST}:${_bop_https:-32169}/ (self-signed certificate; on the first visit,"
  echo "           create the first account with the code that this prints:"
  echo "           sudo -u ${BOP_URL_USER} $2 --config $1 setup-code)"
  echo "Blue Iris: http://${BOP_URL_IRIS_HOST}:${_bop_port:-32168} (plain HTTP, no account)"
}
