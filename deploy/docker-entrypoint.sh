#!/bin/sh
# Container entrypoint (Dockerfile ENTRYPOINT). Needs no network: it only checks the writable
# volumes, seeds the config on first start and execs the service with the arguments it was given.
set -eu

CONFIG_DIR=/app/config
CONFIG="$CONFIG_DIR/blue_onyx_prism_config.json"
uid=$(id -u)
gid=$(id -g)

# The volumes the service writes: the config (written back, plus benchmark.json beside it), the
# compiled-model cache and the data root (`download_dir`: UI downloads, benchmark sets, exports).
# A bind mount that docker created on the host is owned by root, which this non-root user cannot
# write.
not_writable() {
    echo "blue-onyx-prism: $1 is not writable by the container user (uid $uid, gid $gid)." >&2
    echo "Most likely the host directory mounted there is owned by another user (docker creates missing bind-mount directories as root)." >&2
    echo "Fix: on the host run 'sudo chown -R $uid:$gid <host dir>' for that mount, or set 'user: \"<uid>:<gid>\"' in docker-compose.yml to the owner of the directory (see README, Docker)." >&2
    exit 1
}
for d in "$CONFIG_DIR" /app/cache; do
    [ -d "$d" ] && [ -w "$d" ] || not_writable "$d"
done
# The data root only matters for downloads from the web UI; warn instead of refusing to start.
if [ -d /app/data ] && [ ! -w /app/data ]; then
    echo "blue-onyx-prism: warning: /app/data is not writable by uid $uid; downloads and exports from the web UI will fail until it is (chown it on the host)." >&2
fi

if [ ! -f "$CONFIG" ]; then
    cp /app/docker-config.json "$CONFIG" || not_writable "$CONFIG_DIR"
    echo "blue-onyx-prism: created $CONFIG from the image default; edit it and restart the container to change settings."
fi

# Configs created before the image had a read-only root filesystem use the image dir (/app, not
# writable) as the data root.
if ! grep -Eq '"download_dir"[[:space:]]*:[[:space:]]*"' "$CONFIG"; then
    echo "blue-onyx-prism: warning: $CONFIG sets no download_dir, so downloads from the web UI would go to /app, which is read-only; add \"download_dir\": \"/app/data\" to it." >&2
fi

exec /app/blue-onyx-prism --config "$CONFIG" --cache-dir /app/cache "$@"
