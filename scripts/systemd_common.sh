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
