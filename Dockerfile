# Blue Onyx Prism: Linux x86_64 image with Intel GPU support via /dev/dri.
# Build:  docker build -t blue-onyx-prism .
# Run:    docker run --rm -p 127.0.0.1:32168:32168 --device /dev/dri:/dev/dri \
#           --read-only --tmpfs /tmp -v bop-models:/app/models -v bop-data:/app/data \
#           -v $PWD/cache:/app/cache -v $PWD/config:/app/config \
#           --group-add $(getent group render | cut -d: -f3) blue-onyx-prism
# The image user (`openvino`) must be able to write ./cache and ./config: create them yourself
# before the first run (docker creates missing ones as root), or chown them to the uid that
# `docker run --rm --entrypoint id blue-onyx-prism` prints. The entrypoint refuses to start, naming
# the uid, when they are not writable.
# A named volume for /app/models starts with the models baked in at build time; a bind mount
# (e.g. -v $PWD/models:/app/models) replaces them with your own. `-p 127.0.0.1:...` keeps the port
# on this host; for a Blue Iris on another machine use `-p 32168:32168` (see README, Docker).

# Base images are pinned by digest, the tag next to it says what it was; dependabot moves the
# digests (.github/dependabot.yml). Multi-arch index of rust:1-bookworm, read 2026-11-06.
FROM rust:1-bookworm@sha256:e6f2cff35d0f309acf8433b4f05878ed77092602c5065cc81801177da0bb3a9c AS builder
WORKDIR /src
COPY . .
RUN cargo build --release --locked

# Builder glibc 2.36 (bookworm) <= runtime glibc 2.39 (ubuntu24), so the binary runs unchanged.
# Keep the tag equal to OPENVINO_VERSION in src/resources/catalog.rs, and update the digest with it.
# openvino/ubuntu24_runtime ships the OpenVINO runtime, Intel compute runtime (OpenCL) and sets
# INTEL_OPENVINO_DIR, which openvino-finder honours.
# Manifest of openvino/ubuntu24_runtime:2026.4.0 (amd64 only), read 2026-11-06.
FROM openvino/ubuntu24_runtime:2026.4.0@sha256:b326dcae9b91fa25f794a391d9953e65a65096597237e6de431116b184a01718
USER root
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY --from=builder /src/target/release/blue-onyx-prism /app/blue-onyx-prism
COPY --from=builder /src/target/release/blue-onyx-prism-benchmark /app/blue-onyx-prism-benchmark
# ONNX Runtime (CPU flavor, pinned in src/resources/catalog.rs) so `ort:cpu` works; OpenVINO comes
# from the base image. NVIDIA/CUDA is not supported by this image.
RUN /app/blue-onyx-prism setup-onnxruntime --flavor cpu --dir /app/onnxruntime
# Default config (deploy/docker-config.json): IPcam-general on `auto`, OpenVINO from the base
# image, `auto_download: false` (the container never downloads at run time). Everything it
# needs is fetched here, at build time, so the image works offline. To bake other models, edit
# the config and rebuild, or run `fetch --for-config` once with network.
COPY deploy/docker-config.json /app/docker-config.json
COPY deploy/docker-entrypoint.sh /app/docker-entrypoint.sh
RUN /app/blue-onyx-prism --config /app/docker-config.json fetch --for-config \
    && /app/blue-onyx-prism --config /app/docker-config.json list-resources
# /app itself stays root-owned (the binaries, runtimes and default config are not writable by
# the service); only the volume mount points belong to the image user. /app/data is the config's
# `download_dir` (web UI downloads, benchmark datasets, the model-export toolchain).
RUN mkdir -p /app/models /app/cache /app/config /app/data \
    && chown -R openvino:openvino /app/models /app/cache /app/config /app/data \
    && chmod 0755 /app/docker-entrypoint.sh
USER openvino
# glibc gives each thread that allocates its own malloc arena (up to 8 per core); the HTTP,
# worker, collector and runtime threads all touch image-sized buffers, so a long run leaves many
# fragmented arenas that are rarely returned and resident memory creeps up. Two keep it bounded.
ENV MALLOC_ARENA_MAX=2
EXPOSE 32168
# GET /v1/status answers 200 when a model is ready (or degraded) and 503 while models download or
# compile or when none loaded, so `docker ps` shows "starting" until the service can detect. The
# start period covers the first model download + compile (--start-period only suppresses
# failures; a healthy answer ends it early). curl is installed above.
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s --retries=3 \
    CMD curl -fsS -o /dev/null http://127.0.0.1:32168/v1/status || exit 1
# The config lives in the /app/config volume; the entrypoint (deploy/docker-entrypoint.sh, no
# network access) checks that the volumes are writable, copies the image's default config there on
# first start and execs the service with any extra arguments.
ENTRYPOINT ["/bin/sh", "/app/docker-entrypoint.sh"]
