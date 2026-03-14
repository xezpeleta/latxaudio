#!/usr/bin/env bash
# docker-build.sh - Build latxaudio inside Docker and extract artifacts.

set -euo pipefail

TARGET="cpu"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --target) TARGET="$2"; shift 2 ;;
    --target=*) TARGET="${1#*=}"; shift ;;
    *) echo "Unknown argument: $1"; exit 1 ;;
  esac
done

if [[ "$TARGET" != "cpu" && "$TARGET" != "gpu" ]]; then
  echo "Error: --target must be 'cpu' or 'gpu'." >&2
  exit 1
fi

IMAGE="latxaudio-build-${TARGET}"
OUT_DIR="dist/${TARGET}"

echo "==> Building latxaudio (${TARGET^^}) with Docker..."
echo "    Dockerfile : Dockerfile"
echo "    Target     : ${TARGET}"
echo "    Output dir : ${OUT_DIR}"
echo

docker build \
  --target export \
  --build-arg TARGET="${TARGET}" \
  -t "${IMAGE}" \
  .

echo
echo "==> Extracting artifacts to ${OUT_DIR}/ ..."
mkdir -p "${OUT_DIR}"

CID=$(docker create "${IMAGE}")
trap 'docker rm -f "$CID" >/dev/null 2>&1' EXIT

docker cp "${CID}:/dist/latxaudio" "${OUT_DIR}/latxaudio"
docker cp "${CID}:/dist/lib" "${OUT_DIR}/lib"
chmod +x "${OUT_DIR}/latxaudio"

echo
echo "==> Done! Binary and libraries written to ${OUT_DIR}/"
echo "    Run with: ./run.sh"
