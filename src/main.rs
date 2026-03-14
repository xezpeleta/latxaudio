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
use crossterm::event::{self, Event, KeyCode, KeyEvent, KeyEventKind, KeyModifiers};
use crossterm::execute;
use crossterm::terminal::{
    disable_raw_mode, enable_raw_mode, EnterAlternateScreen, LeaveAlternateScreen,
};
use llm::LlmClient;
use ratatui::prelude::*;
use ratatui::widgets::{Block, Borders, Paragraph, Wrap};
use std::fs;
use std::io::{stdout, Stdout};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, Receiver, TryRecvError};
use std::sync::Arc;
use std::thread;
use std::time::{Duration, Instant};
use stt::{SttConfig, SttEngine, SttUpdate};
use tts::{PlayerKind, TtsConfig, TtsEngine};

const DEFAULT_SYSTEM_PROMPT_EU: &str = r#"Audio bidezko elkarrizketa batean zaude, euskaraz.

Testua STT transkripziotik dator eta akatsak egon daitezke (hitz okerrak, etenak, anbiguotasuna).
Interpretatu intentzio probableena eta erantzun lagungarri.

<IMPORTANT>
Debekatuta daude emotikonoak, emojiak eta apaingarri tipografikoak.
Debekatuta daude, gainera, markdown formatu ikurrak (adibidez: *, _, `, #, >, |, ~, [, ]).
Ez erabili inoiz ikur edo formatu horiek erantzunean.
Debekatuta daude neurri-unitateen laburdurak erantzunetan (bereziki: m/s, km/h, km, kg, cm, mm).
Idatzi beti unitateak hitzez eta lokuziorako prest (adibidez: metro segunduko, kilometro orduko, kilometro, kilogramo).
</IMPORTANT>

Erantzun arauak:
- Beti euskaraz.
- Labur eta zuzenean; lehenengo esaldian ideia nagusia.
- Ahots bidez irakurriko da: esaldi naturalak eta erraz ahoskatzekoak.
- Inoiz ez erabili neurri-unitateen laburdurak; beti hitzez eman (adibidez: metro segunduko, kilometro orduko, kilometro, kilogramo).
- Inoiz ez erabili emotikonoak edo emojiak.
- Ez erabili markdown formaturik edo markdown ikurrik.
- Ez zerrenda luzeak edo azalpen gehiegi, erabiltzaileak eskatu ezean.
- Zalantza handia badago transkripzioan, egin galdera labur bakarra argitzeko."#;

const MIC_SETTLE_MS: u64 = 30;
const MAX_UI_MESSAGES: usize = 300;
const NEW_CHAT_BANNER: &str = "Txat berria hasi da. Hitz egin edo idatzi mezua eta sakatu Enter.";
const ECHO_GUARD_WINDOW: Duration = Duration::from_secs(12);
const ECHO_GUARD_MIN_CHARS: usize = 10;

#[derive(Copy, Clone, Debug, Eq, PartialEq)]
enum DuplexState {
    Listening,
    Thinking,
    Speaking,
}

impl DuplexState {
    fn label(self) -> &'static str {
        match self {
            Self::Listening => "LISTENING",
            Self::Thinking => "THINKING",
            Self::Speaking => "SPEAKING",
        }
    }

    fn color(self) -> Color {
        match self {
            Self::Listening => Color::Green,
            Self::Thinking => Color::Yellow,
            Self::Speaking => Color::Blue,
        }
    }
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

#[derive(Copy, Clone)]
enum MessageRole {
    User,
    Assistant,
}

struct UiMessage {
    role: MessageRole,
    content: String,
}

struct AppState {
    duplex_state: DuplexState,
    input: String,
    input_from_stt: bool,
    messages: Vec<UiMessage>,
    streaming_assistant: String,
    assistant_active: bool,
    last_error: Option<String>,
    mic_name: String,
    openai_base_url: String,
    last_assistant_text: String,
    last_assistant_at: Option<Instant>,
}

impl AppState {
    fn new(mic_name: String, openai_base_url: String) -> Self {
        Self {
            duplex_state: DuplexState::Listening,
            input: String::new(),
            input_from_stt: false,
            messages: Vec::new(),
            streaming_assistant: String::new(),
            assistant_active: false,
            last_error: None,
            mic_name,
            openai_base_url,
            last_assistant_text: String::new(),
            last_assistant_at: None,
        }
    }

    fn push_message(&mut self, role: MessageRole, content: String) {
        if content.trim().is_empty() {
            return;
        }
        self.messages.push(UiMessage { role, content });
        if self.messages.len() > MAX_UI_MESSAGES {
            let extra = self.messages.len() - MAX_UI_MESSAGES;
            self.messages.drain(..extra);
        }
    }

    fn apply_stt_partial(&mut self, partial: String) {
        if self.input_from_stt || self.input.trim().is_empty() {
            self.input = partial;
            self.input_from_stt = true;
        }
    }

    fn clear_input(&mut self) {
        self.input.clear();
        self.input_from_stt = false;
    }

    fn reset_chat(&mut self) {
        self.clear_input();
        self.messages.clear();
        self.streaming_assistant.clear();
        self.assistant_active = false;
        self.last_error = None;
        self.last_assistant_text.clear();
        self.last_assistant_at = None;
        self.push_message(MessageRole::Assistant, NEW_CHAT_BANNER.to_string());
    }
}

enum AssistantEvent {
    SpeakingStarted,
    Delta(String),
    Completed(String),
    Failed(String),
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

    let llm = Arc::new(LlmClient::new(
        args.openai_base_url.clone(),
        args.openai_api_key.clone(),
        args.model.clone(),
        args.temperature,
        args.llm_timeout_secs,
    )?);

    let tts_engine = Arc::new(TtsEngine::new(TtsConfig {
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
    })?);

    let system_prompt = resolve_system_prompt(&args)?;
    let mut convo = Conversation::new(system_prompt, args.max_turns);

    let host = cpal::default_host();
    let device = select_input_device(&host, args.device)?;
    let mic_name = device.name()?;

    let (tx, rx_audio) = mpsc::sync_channel::<Vec<f32>>(256);
    let capture_enabled = Arc::new(AtomicBool::new(true));
    let audio_stream = build_audio_stream(&device, tx, Arc::clone(&capture_enabled))?;
    let (audio_stream, capture_rate) = audio_stream;
    audio_stream
        .play()
        .context("Failed to start microphone stream")?;

    let mut app = AppState::new(mic_name, args.openai_base_url.clone());
    app.push_message(
        MessageRole::Assistant,
        "Prest nago. Hitz egin edo idatzi mezua eta sakatu Enter.".to_string(),
    );

    let mut terminal = setup_terminal()?;
    let run_result = run_app(
        &mut terminal,
        &args,
        &stop,
        &mut app,
        &mut convo,
        &mut stt,
        llm,
        tts_engine,
        capture_rate,
        &capture_enabled,
        &audio_stream,
        &rx_audio,
    );
    let restore_result = restore_terminal(&mut terminal);
    drop(audio_stream);

    if let Err(e) = restore_result {
        eprintln!("[latxaudio] Failed to restore terminal: {e}");
    }

    run_result
}

#[allow(clippy::too_many_arguments)]
fn run_app(
    terminal: &mut Terminal<CrosstermBackend<Stdout>>,
    args: &Args,
    stop: &Arc<AtomicBool>,
    app: &mut AppState,
    convo: &mut Conversation,
    stt: &mut SttEngine,
    llm: Arc<LlmClient>,
    tts_engine: Arc<TtsEngine>,
    capture_rate: u32,
    capture_enabled: &Arc<AtomicBool>,
    audio_stream: &cpal::Stream,
    rx_audio: &Receiver<Vec<f32>>,
) -> Result<()> {
    let mut assistant_events: Option<Receiver<AssistantEvent>> = None;
    let mut pending_user_turn: Option<String> = None;

    loop {
        if stop.load(Ordering::SeqCst) {
            break;
        }

        while let Ok(chunk) = rx_audio.try_recv() {
            if app.duplex_state != DuplexState::Listening {
                continue;
            }

            let resampled = resample(&chunk, capture_rate, TARGET_SAMPLE_RATE);
            let SttUpdate {
                partial,
                final_text,
            } = stt.accept_audio(&resampled);

            if let Some(partial) = partial {
                app.apply_stt_partial(partial);
            }

            if let Some(final_text) = final_text {
                if assistant_events.is_some() {
                    continue;
                }

                let send_text = if app.input.trim().is_empty() {
                    final_text.trim().to_string()
                } else {
                    app.input.trim().to_string()
                };

                if send_text.is_empty() {
                    continue;
                }

                if is_probable_tts_echo(app, &send_text) {
                    app.clear_input();
                    app.last_error = Some(
                        "Ignored probable speaker echo. Use headphones or lower speaker volume."
                            .to_string(),
                    );
                    continue;
                }

                assistant_events = Some(begin_assistant_turn(
                    send_text.clone(),
                    app,
                    convo,
                    &llm,
                    &tts_engine,
                    capture_enabled,
                    audio_stream,
                    stt,
                    rx_audio,
                    args.debug,
                ));
                pending_user_turn = Some(send_text);
            }
        }

        let mut completed_text: Option<String> = None;
        let mut failed_text: Option<String> = None;

        if let Some(worker_rx) = assistant_events.as_ref() {
            loop {
                match worker_rx.try_recv() {
                    Ok(event) => match event {
                        AssistantEvent::SpeakingStarted => {
                            set_duplex_state(&mut app.duplex_state, DuplexState::Speaking, args.debug);
                        }
                        AssistantEvent::Delta(delta) => {
                            app.streaming_assistant.push_str(&delta);
                        }
                        AssistantEvent::Completed(text) => {
                            completed_text = Some(text);
                            break;
                        }
                        AssistantEvent::Failed(msg) => {
                            failed_text = Some(msg);
                            break;
                        }
                    },
                    Err(TryRecvError::Empty) => break,
                    Err(TryRecvError::Disconnected) => {
                        failed_text = Some("Assistant worker disconnected unexpectedly".to_string());
                        break;
                    }
                }
            }
        }

        if let Some(err) = failed_text {
            app.last_error = Some(err);
            finish_assistant_turn(
                app,
                convo,
                &mut pending_user_turn,
                None,
                capture_enabled,
                audio_stream,
                stt,
                rx_audio,
                args.post_tts_mute_ms,
                args.debug,
            );
            assistant_events = None;
        }

        if let Some(text) = completed_text {
            finish_assistant_turn(
                app,
                convo,
                &mut pending_user_turn,
                Some(text),
                capture_enabled,
                audio_stream,
                stt,
                rx_audio,
                args.post_tts_mute_ms,
                args.debug,
            );
            assistant_events = None;
        }

        terminal.draw(|frame| render_ui(frame, app))?;

        if event::poll(Duration::from_millis(25))? {
            if let Event::Key(key) = event::read()? {
                if key.kind != KeyEventKind::Press {
                    continue;
                }

                if handle_key_event(key, app, assistant_events.is_some()) {
                    break;
                }

                if key.code == KeyCode::Char('n')
                    && key.modifiers.contains(KeyModifiers::CONTROL)
                {
                    if assistant_events.is_some() {
                        app.last_error = Some(
                            "Assistant is still responding. Wait, then start a new chat."
                                .to_string(),
                        );
                        continue;
                    }

                    convo.clear();
                    pending_user_turn = None;
                    stt.reset();
                    app.reset_chat();
                    continue;
                }

                if key.code == KeyCode::Enter {
                    let text = app.input.trim().to_string();
                    if text.is_empty() {
                        continue;
                    }

                    if is_probable_tts_echo(app, &text) {
                        app.clear_input();
                        app.last_error = Some(
                            "Ignored probable speaker echo. Use headphones or lower speaker volume."
                                .to_string(),
                        );
                        continue;
                    }

                    if assistant_events.is_some() {
                        app.last_error =
                            Some("Assistant is still responding. Wait a moment.".to_string());
                        continue;
                    }

                    assistant_events = Some(begin_assistant_turn(
                        text.clone(),
                        app,
                        convo,
                        &llm,
                        &tts_engine,
                        capture_enabled,
                        audio_stream,
                        stt,
                        rx_audio,
                        args.debug,
                    ));
                    pending_user_turn = Some(text);
                }
            }
        }
    }

    Ok(())
}

fn handle_key_event(key: KeyEvent, app: &mut AppState, assistant_busy: bool) -> bool {
    match key.code {
        KeyCode::Char('c') if key.modifiers.contains(KeyModifiers::CONTROL) => true,
        KeyCode::Char('q') if key.modifiers.is_empty() => true,
        KeyCode::Esc => {
            app.clear_input();
            false
        }
        KeyCode::Backspace => {
            app.input.pop();
            app.input_from_stt = false;
            false
        }
        KeyCode::Enter => false,
        KeyCode::Char(ch) if key.modifiers.is_empty() || key.modifiers == KeyModifiers::SHIFT => {
            let _ = assistant_busy;
            app.input.push(ch);
            app.input_from_stt = false;
            false
        }
        _ => false,
    }
}

#[allow(clippy::too_many_arguments)]
fn begin_assistant_turn(
    user_text: String,
    app: &mut AppState,
    convo: &Conversation,
    llm: &Arc<LlmClient>,
    tts_engine: &Arc<TtsEngine>,
    capture_enabled: &Arc<AtomicBool>,
    audio_stream: &cpal::Stream,
    stt: &mut SttEngine,
    rx_audio: &Receiver<Vec<f32>>,
    debug: bool,
) -> Receiver<AssistantEvent> {
    app.last_error = None;
    app.push_message(MessageRole::User, user_text.clone());
    app.streaming_assistant.clear();
    app.clear_input();
    app.assistant_active = true;

    set_duplex_state(&mut app.duplex_state, DuplexState::Thinking, debug);
    enter_assistant_turn(capture_enabled, audio_stream, stt, rx_audio);

    let messages = convo.build_messages(&user_text);
    let (tx, rx) = mpsc::channel::<AssistantEvent>();
    let llm = Arc::clone(llm);
    let tts_engine = Arc::clone(tts_engine);

    thread::spawn(move || {
        let mut speaking_started = false;
        let mut cleaned_full = String::new();

        match tts_engine.start_session() {
            Ok(mut tts_session) => {
                let stream_result = llm.stream_chat(&messages, |delta| {
                    let clean = strip_visual_decorations(delta);
                    if clean.is_empty() {
                        return Ok(());
                    }

                    if !speaking_started {
                        speaking_started = true;
                        let _ = tx.send(AssistantEvent::SpeakingStarted);
                    }

                    cleaned_full.push_str(&clean);
                    let _ = tx.send(AssistantEvent::Delta(clean.clone()));
                    tts_session.push_text(&clean)
                });

                let finish_result = tts_session.finish();

                match (stream_result, finish_result) {
                    (Ok(_), Ok(())) => {
                        let _ = tx.send(AssistantEvent::Completed(cleaned_full));
                    }
                    (Err(e), _) => {
                        let _ = tx.send(AssistantEvent::Failed(format!("[llm] streaming failed: {e}")));
                    }
                    (_, Err(e)) => {
                        let _ = tx.send(AssistantEvent::Failed(format!("[tts] playback failed: {e}")));
                    }
                }
            }
            Err(e) => {
                let _ = tx.send(AssistantEvent::Failed(format!(
                    "[tts] failed to start session: {e}"
                )));
            }
        }
    });

    rx
}

#[allow(clippy::too_many_arguments)]
fn finish_assistant_turn(
    app: &mut AppState,
    convo: &mut Conversation,
    pending_user_turn: &mut Option<String>,
    assistant_text: Option<String>,
    capture_enabled: &Arc<AtomicBool>,
    audio_stream: &cpal::Stream,
    stt: &mut SttEngine,
    rx_audio: &Receiver<Vec<f32>>,
    post_tts_mute_ms: u64,
    debug: bool,
) {
    if let Some(mut final_text) = assistant_text {
        if !app.streaming_assistant.trim().is_empty() {
            final_text = app.streaming_assistant.clone();
        }

        final_text = final_text.trim().to_string();
        if !final_text.is_empty() {
            app.push_message(MessageRole::Assistant, final_text.clone());
            app.last_assistant_text = final_text.clone();
            app.last_assistant_at = Some(Instant::now());
            if let Some(user_text) = pending_user_turn.take() {
                convo.add_turn(user_text, final_text);
            }
        }
    } else {
        pending_user_turn.take();
    }

    app.streaming_assistant.clear();
    app.assistant_active = false;

    exit_assistant_turn(
        capture_enabled,
        audio_stream,
        stt,
        rx_audio,
        post_tts_mute_ms,
        debug,
    );
    set_duplex_state(&mut app.duplex_state, DuplexState::Listening, debug);
}

fn setup_terminal() -> Result<Terminal<CrosstermBackend<Stdout>>> {
    enable_raw_mode().context("Failed to enable raw mode")?;
    let mut out = stdout();
    execute!(out, EnterAlternateScreen).context("Failed to enter alternate screen")?;
    let backend = CrosstermBackend::new(out);
    let mut terminal = Terminal::new(backend).context("Failed to create terminal")?;
    terminal.clear().context("Failed to clear terminal")?;
    Ok(terminal)
}

fn restore_terminal(terminal: &mut Terminal<CrosstermBackend<Stdout>>) -> Result<()> {
    disable_raw_mode().context("Failed to disable raw mode")?;
    execute!(terminal.backend_mut(), LeaveAlternateScreen)
        .context("Failed to leave alternate screen")?;
    terminal.show_cursor().context("Failed to show cursor")?;
    Ok(())
}

fn render_ui(frame: &mut Frame, app: &AppState) {
    let area = frame.area();
    let chunks = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(6),
            Constraint::Length(3),
            Constraint::Length(3),
        ])
        .split(area);

    render_chat(frame, chunks[0], app);
    render_input(frame, chunks[1], app);
    render_status(frame, chunks[2], app);
}

fn render_chat(frame: &mut Frame, area: Rect, app: &AppState) {
    let mut styled_lines: Vec<Line> = Vec::new();
    let mut plain_lines: Vec<String> = Vec::new();

    for message in &app.messages {
        let (label, color) = match message.role {
            MessageRole::User => ("You", Color::Cyan),
            MessageRole::Assistant => ("Assistant", Color::Green),
        };

        let first = format!("{label}: {}", message.content);
        styled_lines.push(Line::from(vec![
            Span::styled(format!("{label}: "), Style::default().fg(color).bold()),
            Span::styled(message.content.clone(), Style::default().fg(color)),
        ]));
        plain_lines.push(first);
        styled_lines.push(Line::raw(""));
        plain_lines.push(String::new());
    }

    if app.assistant_active {
        if app.streaming_assistant.is_empty() {
            styled_lines.push(Line::from(vec![
                Span::styled("Assistant: ", Style::default().fg(Color::Green).bold()),
                Span::styled("thinking...", Style::default().fg(Color::Yellow)),
            ]));
            plain_lines.push("Assistant: thinking...".to_string());
        } else {
            styled_lines.push(Line::from(vec![
                Span::styled("Assistant: ", Style::default().fg(Color::Green).bold()),
                Span::styled(
                    app.streaming_assistant.clone(),
                    Style::default().fg(Color::Green),
                ),
            ]));
            plain_lines.push(format!("Assistant: {}", app.streaming_assistant));
        }
    }

    let inner_width = area.width.saturating_sub(2) as usize;
    let inner_height = area.height.saturating_sub(2) as usize;
    let total_lines = estimate_wrapped_lines(&plain_lines, inner_width);
    let scroll = total_lines.saturating_sub(inner_height) as u16;

    let chat = Paragraph::new(Text::from(styled_lines))
        .block(
            Block::default()
                .title(" Chat ")
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        )
        .wrap(Wrap { trim: false })
        .scroll((scroll, 0));
    frame.render_widget(chat, area);
}

fn render_input(frame: &mut Frame, area: Rect, app: &AppState) {
    let title = if app.input_from_stt {
        " Input (live STT + typing) "
    } else {
        " Input (type or speak) "
    };

    let input_style = if app.input_from_stt {
        Style::default().fg(Color::Yellow)
    } else {
        Style::default().fg(Color::White)
    };

    let max_visible = area.width.saturating_sub(2) as usize;
    let chars: Vec<char> = app.input.chars().collect();
    let start = chars.len().saturating_sub(max_visible);
    let visible_input: String = chars[start..].iter().collect();
    let cursor_col = visible_input.chars().count() as u16;

    let input = Paragraph::new(visible_input)
        .style(input_style)
        .block(
            Block::default()
                .title(title)
                .borders(Borders::ALL)
                .border_style(Style::default().fg(Color::DarkGray)),
        );
    frame.render_widget(input, area);

    let cursor_x = area.x.saturating_add(1).saturating_add(cursor_col);
    let cursor_y = area.y.saturating_add(1);
    frame.set_cursor_position((cursor_x, cursor_y));
}

fn render_status(frame: &mut Frame, area: Rect, app: &AppState) {
    let mode = Span::styled(
        format!(" {} ", app.duplex_state.label()),
        Style::default()
            .fg(Color::Black)
            .bg(app.duplex_state.color())
            .bold(),
    );

    let mut lines = vec![Line::from(vec![
        Span::raw("Mode: "),
        mode,
        Span::raw("  Mic: "),
        Span::styled(&app.mic_name, Style::default().fg(Color::Cyan)),
    ])];

    lines.push(Line::from(vec![
        Span::raw("Controls: "),
        Span::styled("Enter", Style::default().fg(Color::White).bold()),
        Span::raw(" send  "),
        Span::styled("Ctrl+N", Style::default().fg(Color::White).bold()),
        Span::raw(" new chat  "),
        Span::styled("Esc", Style::default().fg(Color::White).bold()),
        Span::raw(" clear  "),
        Span::styled("q", Style::default().fg(Color::White).bold()),
        Span::raw(" quit"),
    ]));

    if let Some(err) = &app.last_error {
        lines.push(Line::from(vec![
            Span::styled("Error: ", Style::default().fg(Color::Red).bold()),
            Span::styled(err, Style::default().fg(Color::Red)),
        ]));
    } else {
        lines.push(Line::from(vec![
            Span::raw("OpenAI URL: "),
            Span::styled(&app.openai_base_url, Style::default().fg(Color::DarkGray)),
        ]));
    }

    let status = Paragraph::new(Text::from(lines)).block(
        Block::default()
            .title(" Status ")
            .borders(Borders::ALL)
            .border_style(Style::default().fg(Color::DarkGray)),
    );
    frame.render_widget(status, area);
}

fn estimate_wrapped_lines(lines: &[String], width: usize) -> usize {
    if width == 0 {
        return lines.len();
    }

    lines
        .iter()
        .map(|line| {
            let count = line.chars().count();
            let wrapped = count / width;
            if count % width == 0 {
                wrapped.max(1)
            } else {
                wrapped + 1
            }
        })
        .sum()
}

fn is_probable_tts_echo(app: &AppState, candidate: &str) -> bool {
    if candidate.chars().count() < ECHO_GUARD_MIN_CHARS {
        return false;
    }

    let Some(last_at) = app.last_assistant_at else {
        return false;
    };
    if last_at.elapsed() > ECHO_GUARD_WINDOW {
        return false;
    }

    if app.last_assistant_text.trim().is_empty() {
        return false;
    }

    let candidate_norm = normalize_for_echo(candidate);
    let assistant_norm = normalize_for_echo(&app.last_assistant_text);
    if candidate_norm.is_empty() || assistant_norm.is_empty() {
        return false;
    }

    if assistant_norm.contains(&candidate_norm) || candidate_norm.contains(&assistant_norm) {
        return true;
    }

    let candidate_tokens = tokenize_for_echo(&candidate_norm);
    let assistant_tokens = tokenize_for_echo(&assistant_norm);
    if candidate_tokens.is_empty() || assistant_tokens.is_empty() {
        return false;
    }

    let mut overlap = 0usize;
    for token in &candidate_tokens {
        if assistant_tokens.contains(token) {
            overlap += 1;
        }
    }

    let min_len = candidate_tokens.len().min(assistant_tokens.len());
    let overlap_ratio = overlap as f32 / min_len as f32;
    overlap_ratio >= 0.72
}

fn normalize_for_echo(text: &str) -> String {
    text.to_lowercase()
        .chars()
        .map(|ch| {
            if ch.is_alphanumeric() || ch.is_whitespace() {
                ch
            } else {
                ' '
            }
        })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
}

fn tokenize_for_echo(text: &str) -> Vec<String> {
    text.split_whitespace()
        .filter(|t| t.chars().count() >= 2)
        .map(|t| t.to_string())
        .collect()
}

fn strip_visual_decorations(input: &str) -> String {
    let mut cleaned = input.to_string();
    for emoticon in [
        ":)", ":-)", ":(", ":-(", ";)", ";-)", ":D", ":-D", "xD", "XD", ":P", ":-P", ":p",
        ":-p", "^_^", "^^", "T_T", "<3",
    ] {
        cleaned = cleaned.replace(emoticon, "");
    }

    let sanitized: String = cleaned
        .chars()
        .filter(|&c| !is_emoji_or_emoticon(c) && !is_markdown_symbol(c))
        .collect();

    normalize_measurement_units(&sanitized)
}

fn is_emoji_or_emoticon(c: char) -> bool {
    let u = c as u32;
    (0x1F000..=0x1FAFF).contains(&u) || (0x2600..=0x27BF).contains(&u)
}

fn is_markdown_symbol(c: char) -> bool {
    matches!(c, '*' | '_' | '`' | '#' | '>' | '|' | '~' | '[' | ']')
}

fn normalize_measurement_units(input: &str) -> String {
    fn expand_unit_token(token: &str) -> Option<&'static str> {
        match token.to_lowercase().as_str() {
            "m/s" => Some("metro segunduko"),
            "m/s2" | "m/s²" => Some("metro segunduko karratu"),
            "km/h" => Some("kilometro orduko"),
            "km" => Some("kilometro"),
            "kg" => Some("kilogramo"),
            "cm" => Some("zentimetro"),
            "mm" => Some("milimetro"),
            "m2" | "m²" => Some("metro karratu"),
            "m3" | "m³" => Some("metro kubiko"),
            "°c" | "ºc" => Some("gradu celsius"),
            _ => None,
        }
    }

    fn is_unit_char(ch: char) -> bool {
        ch.is_alphanumeric() || matches!(ch, '/' | '°' | 'º' | '²' | '³')
    }

    let mut out = String::with_capacity(input.len());
    let mut token = String::new();

    for ch in input.chars() {
        if is_unit_char(ch) {
            token.push(ch);
            continue;
        }

        if !token.is_empty() {
            if let Some(expanded) = expand_unit_token(&token) {
                out.push_str(expanded);
            } else {
                out.push_str(&token);
            }
            token.clear();
        }

        out.push(ch);
    }

    if !token.is_empty() {
        if let Some(expanded) = expand_unit_token(&token) {
            out.push_str(expanded);
        } else {
            out.push_str(&token);
        }
    }

    out
}

fn drain_audio_queue(rx: &Receiver<Vec<f32>>) {
    while rx.try_recv().is_ok() {}
}

fn enter_assistant_turn(
    capture_enabled: &Arc<AtomicBool>,
    audio_stream: &cpal::Stream,
    stt: &mut SttEngine,
    rx: &Receiver<Vec<f32>>,
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
    rx: &Receiver<Vec<f32>>,
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
