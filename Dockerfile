# Stage 1: Download sherpa-onnx pre-built shared libraries
FROM debian:bookworm-slim AS sherpa-libs

ARG TARGET=cpu
ARG SHERPA_VERSION=1.12.28

RUN apt-get update && apt-get install -y --no-install-recommends \
        wget ca-certificates bzip2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /libs

RUN set -e; \
    if [ "$TARGET" = "gpu" ]; then \
        TARBALL="sherpa-onnx-v${SHERPA_VERSION}-cuda-12.x-cudnn-9.x-linux-x64-gpu.tar.bz2"; \
        DIRNAME="sherpa-onnx-v${SHERPA_VERSION}-cuda-12.x-cudnn-9.x-linux-x64-gpu"; \
    else \
        TARBALL="sherpa-onnx-v${SHERPA_VERSION}-linux-x64-shared.tar.bz2"; \
        DIRNAME="sherpa-onnx-v${SHERPA_VERSION}-linux-x64-shared"; \
    fi \
    && wget -q "https://github.com/k2-fsa/sherpa-onnx/releases/download/v${SHERPA_VERSION}/${TARBALL}" \
    && tar xf "${TARBALL}" \
    && mv "${DIRNAME}" sherpa-onnx-libs \
    && rm "${TARBALL}"


# Stage 2: Build Rust binary
FROM rust:1.85-bookworm AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        libasound2-dev pkg-config \
    && rm -rf /var/lib/apt/lists/*

COPY --from=sherpa-libs /libs/sherpa-onnx-libs /sherpa-onnx-libs
ENV SHERPA_ONNX_LIB_DIR=/sherpa-onnx-libs/lib

WORKDIR /build

COPY Cargo.toml ./
RUN mkdir -p src && echo "fn main() {}" > src/main.rs \
    && cargo build --release \
    && rm -f target/release/deps/latxaudio* target/release/latxaudio

COPY src ./src
RUN touch src/main.rs && cargo build --release


# Stage 3: export artifacts
FROM debian:bookworm-slim AS export

COPY --from=sherpa-libs /libs/sherpa-onnx-libs/lib /dist/lib
COPY --from=builder /build/target/release/latxaudio /dist/latxaudio
