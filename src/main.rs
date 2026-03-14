mod audio;
mod chat;
mod llm;
mod stt;
mod tts;

use anyhow::{Context, Result};
use audio::{build_audio_stream, resample, select_input_device, TARGET_SAMPLE_RATE};
use chat::Conversation;
use clap::{Parser, ValueEnum};
use cpal::traits::{DeviceTrait, StreamTrait};
use llm::LlmClient;
use std::fs;
use std::io::Write;
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::thread;
use std::time::Duration;
use stt::{SttConfig, SttEngine};
use tts::{PlayerKind, TtsConfig, TtsEngine};

const DEFAULT_SYSTEM_PROMPT_EU: &str = r#"Audio bidezko elkarrizketa batean zaude, euskaraz.

Testua STT transkripziotik dator eta akatsak egon daitezke (hitz okerrak, etenak, anbiguotasuna).
Interpretatu intentzio probableena eta erantzun lagungarri.

<IMPORTANT>
Debekatuta daude emotikonoak, emojiak eta apaingarri tipografikoak.
Debekatuta daude, gainera, markdown formatu ikurrak (adibidez: *, _, `, #, >, |, ~, [, ]).
Ez erabili inoiz ikur edo formatu horiek erantzunean.
</IMPORTANT>

Erantzun arauak:
- Beti euskaraz.
- Labur eta zuzenean; lehenengo esaldian ideia nagusia.
- Ahots bidez irakurriko da: esaldi naturalak eta erraz ahoskatzekoak.
- Inoiz ez erabili emotikonoak edo emojiak.
- Ez erabili markdown formaturik edo markdown ikurrik.
- Ez zerrenda luzeak edo azalpen gehiegi, erabiltzaileak eskatu ezean.
- Zalantza handia badago transkripzioan, egin galdera labur bakarra argitzeko."#;

const MIC_SETTLE_MS: u64 = 30;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum DuplexState {
    Listening,
    Thinking,
    Speaking,
}

#[derive(Copy, Clone, Debug, Eq, PartialEq, ValueEnum)]
enum Playback {
    Aplay,
    PwPlay,
}

#[derive(Parser, Debug)]
#[command(
    name = "latxaudio",
    about = "Half-duplex real-time Basque speech chat: Parakeet STT + OpenAI-compatible LLM stream + Piper TTS",
    version
)]
struct Args {
    #[arg(
        long,
        default_value = "./models/parakeet-tdt-0.6b-v3-basque-sherpa-onnx"
    )]
    model_dir: PathBuf,

    #[arg(long, default_value = "./models/silero_vad.onnx")]
    vad_model: PathBuf,

    #[arg(long, default_value = "cpu")]
    provider: String,

    #[arg(long, default_value_t = 2)]
    num_threads: i32,

    #[arg(long)]
    device: Option<usize>,

    #[arg(long, default_value_t = 0.5)]
    vad_threshold: f32,

    #[arg(long, default_value_t = 0.18)]
    vad_min_silence: f32,

    #[arg(long, default_value_t = 0.15)]
    vad_min_speech: f32,

    #[arg(long, default_value_t = 5.0)]
    max_speech_duration: f32,

    #[arg(long, default_value_t = false)]
    debug: bool,

    #[arg(long, env = "OPENAI_BASE_URL", default_value = "https://api.openai.com/v1")]
    openai_base_url: String,

    #[arg(short = 'm', long = "model", env = "OPENAI_MODEL", default_value = "gpt-4o-mini")]
    model: String,

    #[arg(long, env = "OPENAI_API_KEY")]
    openai_api_key: Option<String>,

    #[arg(long, default_value_t = 0.2)]
    temperature: f32,

    #[arg(long, default_value_t = 120)]
    llm_timeout_secs: u64,

    #[arg(long, env = "CHAT_SYSTEM_PROMPT", help = "System prompt for the assistant (default enforces Basque responses)")]
    system_prompt: Option<String>,

    #[arg(long, env = "CHAT_SYSTEM_PROMPT_FILE", help = "Path to a text file containing the system prompt")]
    system_prompt_file: Option<PathBuf>,

    #[arg(long, default_value_t = 8)]
    max_turns: usize,

    #[arg(long, env = "PIPER_MODEL", default_value = "./models/piper/eu-maider-medium.onnx")]
    piper_model: PathBuf,

    #[arg(long, env = "PIPER_CONFIG")]
    piper_config: Option<PathBuf>,

    #[arg(long)]
    piper_speaker: Option<i32>,

    #[arg(long, default_value_t = 1.0)]
    piper_length_scale: f32,

    #[arg(long, default_value_t = 0.667)]
    piper_noise_scale: f32,

    #[arg(long, default_value_t = 0.8)]
    piper_noise_w_scale: f32,

    #[arg(long, default_value_t = 0.06)]
    piper_sentence_silence: f32,

    #[arg(long, default_value_t = 900)]
    post_tts_mute_ms: u64,

    #[arg(long, value_enum, default_value_t = Playback::Aplay)]
    player: Playback,
}

fn main() -> Result<()> {
    let args = Args::parse();

    if args.provider != "cpu" && args.provider != "cuda" {
        anyhow::bail!("--provider must be 'cpu' or 'cuda'");
    }

    let stop = Arc::new(AtomicBool::new(false));
    {
        let stop = Arc::clone(&stop);
        ctrlc::set_handler(move || {
            eprintln!("\n[latxaudio] Ctrl+C - stopping...");
            stop.store(true, Ordering::SeqCst);
        })
        .context("Failed to set Ctrl+C handler")?;
    }

    let stt_cfg = SttConfig {
        model_dir: args.model_dir.clone(),
        vad_model: args.vad_model.clone(),
        provider: args.provider.clone(),
        num_threads: args.num_threads,
        vad_threshold: args.vad_threshold,
        vad_min_silence: args.vad_min_silence,
        vad_min_speech: args.vad_min_speech,
        max_speech_duration: args.max_speech_duration,
        debug: args.debug,
    };
    let mut stt = SttEngine::new(&stt_cfg)?;

    let llm = LlmClient::new(
        args.openai_base_url.clone(),
        args.openai_api_key.clone(),
        args.model.clone(),
        args.temperature,
        args.llm_timeout_secs,
    )?;

    let tts_engine = TtsEngine::new(TtsConfig {
        piper_model: args.piper_model.clone(),
        piper_config: args.piper_config.clone(),
        piper_speaker: args.piper_speaker,
        piper_length_scale: args.piper_length_scale,
        piper_noise_scale: args.piper_noise_scale,
        piper_noise_w_scale: args.piper_noise_w_scale,
        piper_sentence_silence: args.piper_sentence_silence,
        player: match args.player {
            Playback::Aplay => PlayerKind::Aplay,
            Playback::PwPlay => PlayerKind::PwPlay,
        },
    })?;

    let system_prompt = resolve_system_prompt(&args)?;
    let mut convo = Conversation::new(system_prompt, args.max_turns);

    let host = cpal::default_host();
    let device = select_input_device(&host, args.device)?;
    println!("Input device : {}", device.name()?);

    let (tx, rx) = mpsc::sync_channel::<Vec<f32>>(256);
    let capture_enabled = Arc::new(AtomicBool::new(true));
    let audio_stream = build_audio_stream(&device, tx, Arc::clone(&capture_enabled))?;
    let (audio_stream, capture_rate) = audio_stream;

    audio_stream
        .play()
        .context("Failed to start microphone stream")?;

    println!(
        "[latxaudio] Ready. Speak now. Half-duplex mode is enabled (mic pauses during assistant response)."
    );
    println!("[latxaudio] Basque-first mode enabled.");
    println!("[latxaudio] OpenAI base URL: {}", args.openai_base_url);
    println!();

    let mut duplex_state = DuplexState::Listening;

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }

        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(chunk) => {
                if duplex_state != DuplexState::Listening {
                    continue;
                }

                let resampled = resample(&chunk, capture_rate, TARGET_SAMPLE_RATE);
                if let Some(user_text) = stt.accept_audio(&resampled) {
                    let user_text = user_text.trim().to_string();
                    if user_text.is_empty() {
                        continue;
                    }

                    println!("\n[user] {user_text}");
                    print!("[assistant] ");
                    std::io::stdout()
                        .flush()
                        .context("Failed to flush stdout")?;

                    set_duplex_state(&mut duplex_state, DuplexState::Thinking, args.debug);
                    enter_assistant_turn(&capture_enabled, &audio_stream, &mut stt, &rx);

                    let messages = convo.build_messages(&user_text);
                    if args.debug {
                        eprintln!("[latxaudio][to-llm] {}", user_text);
                    }

                    match tts_engine.start_session() {
                        Ok(mut tts_session) => {
                            let assistant_result = llm.stream_chat(&messages, |delta| {
                                let clean = strip_visual_decorations(delta);
                                if clean.is_empty() {
                                    return Ok(());
                                }

                                if duplex_state != DuplexState::Speaking {
                                    set_duplex_state(
                                        &mut duplex_state,
                                        DuplexState::Speaking,
                                        args.debug,
                                    );
                                }

                                print!("{clean}");
                                std::io::stdout()
                                    .flush()
                                    .context("Failed to flush streamed output")?;
                                tts_session.push_text(&clean)
                            });

                            let finish_result = tts_session.finish();
                            println!();

                            match (assistant_result, finish_result) {
                                (Ok(assistant_text), Ok(())) => {
                                    convo.add_turn(user_text, assistant_text);
                                }
                                (Err(e), _) => {
                                    eprintln!("[llm] streaming failed: {e}");
                                }
                                (_, Err(e)) => {
                                    eprintln!("[tts] playback failed: {e}");
                                }
                            }
                        }
                        Err(e) => {
                            println!();
                            eprintln!("[tts] failed to start session: {e}");
                        }
                    }

                    exit_assistant_turn(
                        &capture_enabled,
                        &audio_stream,
                        &mut stt,
                        &rx,
                        args.post_tts_mute_ms,
                        args.debug,
                    );
                    set_duplex_state(&mut duplex_state, DuplexState::Listening, args.debug);
                }
            }
            Err(RecvTimeoutError::Timeout) => {}
            Err(RecvTimeoutError::Disconnected) => {
                eprintln!("[latxaudio] Audio stream disconnected.");
                break;
            }
        }
    }

    drop(audio_stream);
    println!("\n[latxaudio] Stopped.");
    Ok(())
}

fn strip_visual_decorations(input: &str) -> String {
    let mut cleaned = input.to_string();
    for emoticon in [
        ":)", ":-)", ":(", ":-(", ";)", ";-)", ":D", ":-D", "xD", "XD", ":P", ":-P", ":p",
        ":-p", "^_^", "^^", "T_T", "<3",
    ] {
        cleaned = cleaned.replace(emoticon, "");
    }

    cleaned
        .chars()
        .filter(|&c| !is_emoji_or_emoticon(c) && !is_markdown_symbol(c))
        .collect()
}

fn is_emoji_or_emoticon(c: char) -> bool {
    let u = c as u32;
    (0x1F000..=0x1FAFF).contains(&u) || (0x2600..=0x27BF).contains(&u)
}

fn is_markdown_symbol(c: char) -> bool {
    matches!(c, '*' | '_' | '`' | '#' | '>' | '|' | '~' | '[' | ']')
}

fn drain_audio_queue(rx: &mpsc::Receiver<Vec<f32>>) {
    while rx.try_recv().is_ok() {}
}

fn enter_assistant_turn(
    capture_enabled: &Arc<AtomicBool>,
    audio_stream: &cpal::Stream,
    stt: &mut SttEngine,
    rx: &mpsc::Receiver<Vec<f32>>,
) {
    capture_enabled.store(false, Ordering::Release);
    if let Err(e) = audio_stream.pause() {
        eprintln!("[audio] warning: failed to pause mic stream: {e}");
    }

    stt.reset();
    drain_audio_queue(rx);
    thread::sleep(Duration::from_millis(MIC_SETTLE_MS));
    drain_audio_queue(rx);
}

fn exit_assistant_turn(
    capture_enabled: &Arc<AtomicBool>,
    audio_stream: &cpal::Stream,
    stt: &mut SttEngine,
    rx: &mpsc::Receiver<Vec<f32>>,
    post_tts_mute_ms: u64,
    debug: bool,
) {
    if post_tts_mute_ms > 0 {
        if debug {
            eprintln!("[audio] post-tts mute cooldown: {} ms", post_tts_mute_ms);
        }
        thread::sleep(Duration::from_millis(post_tts_mute_ms));
    }

    stt.reset();
    drain_audio_queue(rx);
    thread::sleep(Duration::from_millis(MIC_SETTLE_MS));
    drain_audio_queue(rx);

    capture_enabled.store(true, Ordering::Release);
    if let Err(e) = audio_stream.play() {
        eprintln!("[audio] warning: failed to resume mic stream: {e}");
    }
}

fn set_duplex_state(state: &mut DuplexState, next: DuplexState, debug: bool) {
    if *state == next {
        return;
    }

    if debug {
        eprintln!("[duplex] {:?} -> {:?}", *state, next);
    }
    *state = next;
}

fn resolve_system_prompt(args: &Args) -> Result<Option<String>> {
    if args.system_prompt.is_some() && args.system_prompt_file.is_some() {
        anyhow::bail!(
            "Use either --system-prompt or --system-prompt-file, but not both"
        );
    }

    if let Some(path) = &args.system_prompt_file {
        let prompt = fs::read_to_string(path)
            .with_context(|| format!("Failed to read system prompt file: {}", path.display()))?;
        return Ok(Some(prompt));
    }

    Ok(args
        .system_prompt
        .clone()
        .or_else(|| Some(DEFAULT_SYSTEM_PROMPT_EU.to_string())))
}
