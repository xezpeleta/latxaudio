#!/usr/bin/env bash
# scripts/download-models.sh - Download STT model files for latxaudio.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../models"
MODEL_REPO="xezpeleta/parakeet-tdt-0.6b-v3-basque-sherpa-onnx"
MODEL_DIR="${MODELS_DIR}/parakeet-tdt-0.6b-v3-basque-sherpa-onnx"
VAD_URL="https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx"
VAD_FILE="${MODELS_DIR}/silero_vad.onnx"

HF_BASE="https://huggingface.co/${MODEL_REPO}/resolve/main"
MODEL_FILES=(
  "encoder.int8.onnx"
  "decoder.int8.onnx"
  "joiner.int8.onnx"
  "tokens.txt"
)

mkdir -p "${MODEL_DIR}"

download_if_missing() {
  local url="$1"
  local dest="$2"

  if [[ -f "${dest}" ]]; then
    echo "  [skip] $(basename "${dest}") already exists"
  else
    echo "  [dl]   ${url}"
    wget -q --show-progress -O "${dest}" "${url}"
  fi
}

echo "==> Downloading Parakeet TDT 0.6B v3 Basque model..."
for f in "${MODEL_FILES[@]}"; do
  download_if_missing "${HF_BASE}/${f}" "${MODEL_DIR}/${f}"
done

echo
echo "==> Downloading Silero VAD model..."
download_if_missing "${VAD_URL}" "${VAD_FILE}"

echo
echo "==> STT models ready in ${MODELS_DIR}/"
echo "    Model dir : ${MODEL_DIR}"
echo "    VAD model : ${VAD_FILE}"
