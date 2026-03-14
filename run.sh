#!/usr/bin/env bash
# run.sh - Launch latxaudio with bundled shared libraries.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROVIDER="cpu"
EXTRA_ARGS=()

has_arg() {
  local needle="$1"
  for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "$needle" || "$arg" == "$needle="* ]]; then
      return 0
    fi
  done
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --provider) PROVIDER="$2"; shift 2 ;;
    --provider=*) PROVIDER="${1#*=}"; shift ;;
    *) EXTRA_ARGS+=("$1"); shift ;;
  esac
done

case "$PROVIDER" in
  cpu) TARGET="cpu" ;;
  cuda) TARGET="gpu" ;;
  *) echo "Error: --provider must be 'cpu' or 'cuda'." >&2; exit 1 ;;
esac

DIST_DIR="${SCRIPT_DIR}/dist/${TARGET}"
BINARY="${DIST_DIR}/latxaudio"
LIB_DIR="${DIST_DIR}/lib"
MODELS_DIR="${SCRIPT_DIR}/models"
MODEL_DIR="${MODELS_DIR}/parakeet-tdt-0.6b-v3-basque-sherpa-onnx"
VAD_MODEL="${MODELS_DIR}/silero_vad.onnx"
PIPER_DIR="${MODELS_DIR}/piper"
PIPER_MODEL_DEFAULT="${PIPER_DIR}/eu-maider-medium.onnx"
PIPER_CONFIG_DEFAULT="${PIPER_MODEL_DEFAULT}.json"
HELP_ONLY=false

if has_arg "--help" || has_arg "-h"; then
  HELP_ONLY=true
fi

if [[ ! -x "${BINARY}" ]]; then
  echo "Error: ${BINARY} not found or not executable."
  echo "       Run: ./docker-build.sh --target ${TARGET}"
  exit 1
fi

if [[ "${HELP_ONLY}" != "true" ]]; then
  if [[ ! -f "${VAD_MODEL}" || ! -f "${MODEL_DIR}/tokens.txt" ]]; then
    echo "==> STT models not found - running scripts/download-models.sh ..."
    bash "${SCRIPT_DIR}/scripts/download-models.sh"
  fi

  if ! has_arg "--piper-model" && [[ ! -f "${PIPER_MODEL_DEFAULT}" || ! -f "${PIPER_CONFIG_DEFAULT}" ]]; then
    echo "==> Basque Piper voice not found - downloading itzune/maider-tts ..."
    bash "${SCRIPT_DIR}/scripts/download-piper-voice.sh"
  fi
fi

if [[ "${HELP_ONLY}" != "true" ]]; then
  if [[ -z "${OPENAI_API_KEY:-}" ]] && ! has_arg "--openai-api-key"; then
    echo "[warn] OPENAI_API_KEY is not set."
    echo "       This is fine for local OpenAI-compatible servers without auth."
    echo "       For OpenAI cloud, export OPENAI_API_KEY before running."
  fi
fi

export LD_LIBRARY_PATH="${LIB_DIR}${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

exec "${BINARY}" \
  --model-dir "${MODEL_DIR}" \
  --vad-model "${VAD_MODEL}" \
  --provider "${PROVIDER}" \
  "${EXTRA_ARGS[@]}"
