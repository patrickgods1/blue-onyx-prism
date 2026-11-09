# shellcheck shell=sh
# Shared helpers for the systemd install flows: scripts/install_systemd.sh (release archive) and
# the .deb / .rpm maintainer scripts (packaging/linux/). Sourced, never executed. POSIX sh only
# (Debian's /bin/sh is dash); the caller sets its own `set -eu` and must be root.

# Service user (member of render/video for /dev/dri access when those groups exist).
BOP_USER=blueonyx

# bop_check_paths PATH...: the paths end up in a sed expression and in JSON.
bop_check_paths() {
  case "$*" in
    *'#'* | *'"'* | *'\'*) echo "Paths must not contain '#', '\"' or '\\'." >&2; return 1 ;;
  esac
}

# bop_ensure_user DATA_DIR: create the system user if missing (home = data root), then add it to
# render/video when those groups exist.
bop_ensure_user() {
  if ! id -u "$BOP_USER" >/dev/null 2>&1; then
    useradd --system --home-dir "$1" --no-create-home --shell /usr/sbin/nologin "$BOP_USER"
  else
    usermod --home "$1" "$BOP_USER"
  fi
  for g in render video; do
    if getent group "$g" >/dev/null; then usermod -aG "$g" "$BOP_USER"; fi
  done
}

# bop_own_data_root DATA_DIR: an uninstall keeps the data root, so an install can meet files of an
# earlier $BOP_USER with another uid (the user was deleted by hand and made again, or the folder
# came from a backup). `install -d -o` corrects the named folders only and `chown -R` covers only
# models/, so look at everything: one entry (the folder itself or anything in it) of another
# owner or group and the whole tree is chowned. Run it after bop_ensure_user and before
# `install -d`. -h, find's default -P and the `/.`: links are never followed (uv's Python
# environment has some), a data root that is itself a link to a larger disk is. `-print -quit`
# stops at the first hit (GNU and BSD find). A failure warns: the data is intact.
bop_own_data_root() {
  _own="$1"
  [ -d "$_own" ] || return 0
  _first="$(find "$_own/." \( ! -user "$BOP_USER" -o ! -group "$BOP_USER" \) -print -quit 2>/dev/null || true)"
  [ -n "$_first" ] || return 0
  echo "$_own holds files not owned by $BOP_USER (first: $_first): changing the owner of everything in it, which can take a while." >&2
  chown -hR "$BOP_USER:$BOP_USER" "$_own/." \
    || echo "Could not change the owner of everything in $_own, so the service may fail to read it (is it a network share?). Fix it with: sudo chown -hR $BOP_USER:$BOP_USER $_own/." >&2
}

# bop_seed_config INSTALL_DIR DATA_DIR CONFIG WITH_RUNTIMES: write CONFIG unless it exists (never
# overwrites), then fix owner/mode. download_dir and cache_dir always point at the data root.
# WITH_RUNTIMES=1 (archive flow) also names the bundled, root-owned openvino/ and onnxruntime/
# dirs (a runtime named by openvino_dir / onnxruntime_dir is user-managed: never downloaded or
# replaced); 0 (packages) leaves them out so the service downloads runtimes into download_dir.
bop_seed_config() {
  _inst="$1"; _data="$2"; _cfg="$3"; _rt="$4"
  keys="  \"download_dir\": \"$_data\",
  \"cache_dir\": \"$_data/cache\","
  if [ "$_rt" = 1 ]; then
    if [ -d "$_inst/openvino" ]; then
      keys="$keys
  \"openvino_dir\": \"$_inst/openvino\","
    fi
    ort=""
    if [ -f "$_inst/onnxruntime/active.txt" ]; then
      flavor="$(tr -d '[:space:]' < "$_inst/onnxruntime/active.txt")"
      [ -n "$flavor" ] && [ -d "$_inst/onnxruntime/$flavor" ] && ort="$_inst/onnxruntime/$flavor"
    elif ls "$_inst"/onnxruntime/libonnxruntime.so* >/dev/null 2>&1; then
      ort="$_inst/onnxruntime"
    fi
    if [ -n "$ort" ]; then
      keys="$keys
  \"onnxruntime_dir\": \"$ort\","
    fi
  fi

  if [ ! -f "$_cfg" ]; then
    src=""
    if [ -f "$_inst/blue_onyx_prism_config.json" ]; then
      src="$_inst/blue_onyx_prism_config.json"
    fi
    if [ -n "$src" ] && grep -q '"download_dir"' "$src"; then
      cp "$src" "$_cfg"
      echo "Copied $src to $_cfg; it already sets download_dir: check that it points at $_data."
    elif [ -n "$src" ] && [ "$(head -n 1 "$src")" = "{" ]; then
      # Pretty JSON as the service writes it: insert the keys after the opening brace.
      KEYS="$keys" awk 'NR == 1 { print; print ENVIRON["KEYS"]; next } { print }' "$src" > "$_cfg"
      echo "Moved the config to $_cfg (download_dir, cache_dir and runtime dirs added)."
    elif [ -n "$src" ]; then
      cp "$src" "$_cfg"
      echo "Copied $src to $_cfg. Add these keys to it by hand:" >&2
      echo "$keys" >&2
    else
      printf '{\n%s\n  "models_dir": "models"\n}\n' "$keys" > "$_cfg"
    fi
  fi
  chown "$BOP_USER:$BOP_USER" "$_cfg"
  chmod 0640 "$_cfg"
}

# bop_remove_legacy_unit: the pre-rename unit (Blue Onyx OpenVINO) fights over the port.
bop_remove_legacy_unit() {
  if [ -f /etc/systemd/system/blue-onyx-openvino.service ]; then
    systemctl disable --now blue-onyx-openvino 2>/dev/null || true
    rm -f /etc/systemd/system/blue-onyx-openvino.service
  fi
}

# bop_install_unit UNIT_SRC UNIT_DEST INSTALL_DIR: the unit template names /opt/blue-onyx-prism.
bop_install_unit() {
  sed "s#/opt/blue-onyx-prism#$3#g" "$1" > "$2"
}

# bop_print_urls CONFIG EXE: the addresses an install prints. The web UI is HTTPS (self-signed
# certificate, a sign-in; the first visit creates the first account with the one-time code that
# `setup-code` prints); Blue Iris uses the plain HTTP port with no account. The ports are the
# config's when it names them (a pretty config, one key per line), else the defaults.
bop_print_urls() {
  _port=$(sed -n 's/^[[:space:]]*"port":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$1" 2>/dev/null | head -n 1 || true)
  _https=$(sed -n 's/^[[:space:]]*"https_port":[[:space:]]*\([0-9][0-9]*\).*/\1/p' "$1" 2>/dev/null | head -n 1 || true)
  echo "Web UI:    https://<this machine>:${_https:-32169}/ (self-signed certificate; on the first visit,"
  echo "           create the first account with the code that this prints:"
  echo "           sudo -u $BOP_USER $2 --config $1 setup-code)"
  echo "Blue Iris: http://<this machine's IP>:${_port:-32168} (plain HTTP, no account)"
}
