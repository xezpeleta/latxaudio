#!/usr/bin/env bash
# scripts/download-piper-voice.sh - Download Basque Piper voice (itzune/maider-tts).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/../models/piper"
REPO_ID="itzune/maider-tts"
VOICE_ONNX="eu-maider-medium.onnx"
VOICE_CONFIG="eu-maider-medium.onnx.json"

if ! command -v hf >/dev/null 2>&1; then
  echo "Error: hf CLI is not installed or not in PATH." >&2
  echo "Install it from: https://huggingface.co/docs/huggingface_hub/en/guides/cli" >&2
  exit 1
fi

mkdir -p "${MODELS_DIR}"

echo "==> Downloading Basque Piper voice from ${REPO_ID} ..."
hf download "${REPO_ID}" "${VOICE_ONNX}" "${VOICE_CONFIG}" --local-dir "${MODELS_DIR}"

echo
echo "==> Piper voice ready:"
echo "    Model : ${MODELS_DIR}/${VOICE_ONNX}"
echo "    Config: ${MODELS_DIR}/${VOICE_CONFIG}"
