# latxaudio

Real-time Basque voice chat CLI using:

- Parakeet TDT v3 Basque + Silero VAD for speech-to-text (simulated streaming)
- OpenAI-compatible Chat Completions streaming (`stream=true`)
- Piper for low-latency text-to-speech playback

Language setup is Basque-first end-to-end: Basque ASR (Parakeet), Basque TTS (Maider Piper voice), and Basque default system prompt for the LLM.

Default system prompt is tuned for voice chat: it assumes possible STT errors and asks for concise, direct, speech-friendly Basque replies (no emojis/emoticons).
Additionally, streamed assistant text is sanitized at runtime to remove common emoji/emoticon symbols before printing and TTS.

`latxaudio` runs in **half-duplex** mode: it listens while you speak, then pauses the mic while the assistant responds (both text stream and TTS), then resumes listening.

## Features

- Real-time microphone transcription with partial updates
- Final utterance detection with Silero VAD
- Streaming assistant tokens printed immediately
- Incremental TTS chunking to start playback before full response is finished
- OpenAI default endpoint with configurable base URL for local/remote compatible servers

## Requirements

- Linux
- `piper` installed and available in `PATH`
- `aplay` (default) or `pw-play`
- OpenAI-compatible API server
- STT models (Parakeet + Silero VAD)
- Piper voice model (`.onnx`) and matching config (`.onnx.json`)

## Build

### Option A: Docker build (recommended)

```bash
./docker-build.sh
```

This creates:

```text
dist/cpu/latxaudio
dist/cpu/lib/
```

GPU libraries can be built with:

```bash
./docker-build.sh --target gpu
```

### Option B: Local Cargo

If you have Rust installed and `sherpa-onnx` shared libs available:

```bash
cargo build --release
```

## Download STT and TTS models

```bash
./scripts/download-models.sh
./scripts/download-piper-voice.sh
```

## Quick start

Set environment variables:

```bash
export OPENAI_API_KEY="<your-key>"
export OPENAI_MODEL="gpt-4o-mini"
export OPENAI_BASE_URL="https://api.openai.com/v1"
```

Default Piper voice is downloaded to:

- `./models/piper/eu-maider-medium.onnx`
- `./models/piper/eu-maider-medium.onnx.json`

You can override with custom voice files if needed:

```bash
export PIPER_MODEL="/absolute/path/to/voice.onnx"
export PIPER_CONFIG="/absolute/path/to/voice.onnx.json"
```

Run:

```bash
./run.sh
```

Use a custom OpenAI-compatible endpoint:

```bash
OPENAI_BASE_URL="http://localhost:11434/v1" ./run.sh --model llama3.1
```

If your local server does not require auth, you can run without `OPENAI_API_KEY`.

Choose playback backend:

```bash
./run.sh --player pw-play
```

Use a system prompt from a text file:

```bash
./run.sh --system-prompt-file ./prompts/system.txt
```

## Important flags

- `--openai-base-url` (default: `https://api.openai.com/v1`)
- `--model` / `-m`
- `--openai-api-key`
- `--system-prompt`
- `--system-prompt-file`
- `--piper-model`
- `--piper-config`
- `--provider cpu|cuda`
- `--device <index>`
- `--vad-threshold <float>`
- `--vad-min-silence <seconds>` (default `0.18`)
- `--vad-min-speech <seconds>` (default `0.15`)
- `--max-speech-duration <seconds>`
- `--post-tts-mute-ms <milliseconds>` (default `900`)
- `--max-turns <n>`

Get full help:

```bash
./run.sh --help
```
