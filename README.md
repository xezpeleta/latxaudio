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

- Full-screen TUI with dedicated chat, input, and status panels
- Clear user/assistant separation with different colors in chat history
- Real-time microphone transcription that is injected into the editable input box
- Manual typing support in the same input box before sending to LLM
- Live state indicator: listening, thinking, speaking
- Final utterance detection with Silero VAD
- Streaming assistant tokens rendered immediately in chat while TTS plays
- Incremental TTS chunking to start playback before full response is finished
- OpenAI default endpoint with configurable base URL for local/remote compatible servers
- `Translate` function (`Ctrl+T`) with target language picker in a popup window
- Translate mode: microphone Basque speech is auto-translated by LLM and synthesized via Piper TTS
- Translation targets are enabled only when both matching Piper files exist in `models/piper` (`*.onnx` and `*.onnx.json`)

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

When running, the TUI opens in alternate-screen mode with three areas:

- Chat panel: conversation history and live assistant stream
- Input panel: live STT draft + editable typing area
- Status panel: mode badge and runtime hints

Keyboard controls:

- `Enter`: send current input to the assistant
- `Ctrl+N`: start a new chat (clears current conversation and input)
- `Ctrl+T`: open translation target picker and switch function mode
- `Esc`: clear current input
- `q` or `Ctrl+C`: quit

When Translate mode is enabled, the app is no longer a chat: each final Basque utterance from the microphone is translated directly to the selected target language (no chat history memory), streamed on screen, and spoken by Piper.
Each enabled language uses its own Piper voice model+config from `models/piper`.
Languages without a detected voice, or without the matching `.onnx.json`, are shown as disabled in the picker and cannot be selected.
Use `Ctrl+T` again and choose `Back to chat mode` to return to normal chat behavior.

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
