use anyhow::{Context, Result};
use std::fs;
use std::io::Write;
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};

#[derive(Clone, Copy)]
pub enum PlayerKind {
    Aplay,
    PwPlay,
}

pub struct TtsConfig {
    pub piper_model: PathBuf,
    pub piper_config: Option<PathBuf>,
    pub piper_speaker: Option<i32>,
    pub piper_length_scale: f32,
    pub piper_noise_scale: f32,
    pub piper_noise_w_scale: f32,
    pub piper_sentence_silence: f32,
    pub player: PlayerKind,
}

pub struct TtsEngine {
    cfg: TtsConfig,
    sample_rate: u32,
}

impl TtsEngine {
    pub fn new(cfg: TtsConfig) -> Result<Self> {
        if !cfg.piper_model.exists() {
            anyhow::bail!(
                "Piper model file not found: {}",
                cfg.piper_model.to_string_lossy()
            );
        }

        let config_path = config_path_for_model(&cfg.piper_model, cfg.piper_config.as_deref());
        let sample_rate = read_sample_rate_from_config(&config_path).unwrap_or(22_050);

        Ok(Self { cfg, sample_rate })
    }

    pub fn start_session(&self) -> Result<TtsSession> {
        let mut piper = Command::new("piper");
        piper
            .arg("--model")
            .arg(&self.cfg.piper_model)
            .arg("--output-raw")
            .arg("--length-scale")
            .arg(self.cfg.piper_length_scale.to_string())
            .arg("--noise-scale")
            .arg(self.cfg.piper_noise_scale.to_string())
            .arg("--noise-w-scale")
            .arg(self.cfg.piper_noise_w_scale.to_string())
            .arg("--sentence-silence")
            .arg(self.cfg.piper_sentence_silence.to_string())
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::inherit());

        if let Some(config) = &self.cfg.piper_config {
            piper.arg("--config").arg(config);
        }

        if let Some(speaker) = self.cfg.piper_speaker {
            piper.arg("--speaker").arg(speaker.to_string());
        }

        let mut piper_child = piper.spawn().context("Failed to start piper process")?;
        let piper_stdout = piper_child
            .stdout
            .take()
            .context("Failed to capture piper stdout")?;

        let mut player = match self.cfg.player {
            PlayerKind::Aplay => {
                let mut cmd = Command::new("aplay");
                cmd.arg("-q")
                    .arg("-t")
                    .arg("raw")
                    .arg("-f")
                    .arg("S16_LE")
                    .arg("-r")
                    .arg(self.sample_rate.to_string())
                    .arg("-c")
                    .arg("1");
                cmd
            }
            PlayerKind::PwPlay => {
                let mut cmd = Command::new("pw-play");
                cmd.arg("--rate")
                    .arg(self.sample_rate.to_string())
                    .arg("--channels")
                    .arg("1")
                    .arg("--format")
                    .arg("s16")
                    .arg("-");
                cmd
            }
        };

        player
            .stdin(Stdio::from(piper_stdout))
            .stdout(Stdio::null())
            .stderr(Stdio::inherit());

        let player_child = player
            .spawn()
            .context("Failed to start playback process (aplay/pw-play)")?;

        let piper_stdin = piper_child
            .stdin
            .take()
            .context("Failed to capture piper stdin")?;

        Ok(TtsSession {
            piper_stdin: Some(piper_stdin),
            piper_child: Some(piper_child),
            player_child: Some(player_child),
            chunker: SentenceChunker::new(120),
        })
    }
}

pub struct TtsSession {
    piper_stdin: Option<ChildStdin>,
    piper_child: Option<Child>,
    player_child: Option<Child>,
    chunker: SentenceChunker,
}

impl TtsSession {
    pub fn push_text(&mut self, text: &str) -> Result<()> {
        for chunk in self.chunker.push(text) {
            self.send_chunk(&chunk)?;
        }
        Ok(())
    }

    pub fn finish(&mut self) -> Result<()> {
        if let Some(last) = self.chunker.finish() {
            self.send_chunk(&last)?;
        }

        if let Some(mut stdin) = self.piper_stdin.take() {
            stdin.flush().ok();
            drop(stdin);
        }

        if let Some(mut piper_child) = self.piper_child.take() {
            let status = piper_child.wait().context("Failed waiting for piper")?;
            if !status.success() {
                anyhow::bail!("Piper process exited with status {status}");
            }
        }

        if let Some(mut player_child) = self.player_child.take() {
            let status = player_child
                .wait()
                .context("Failed waiting for playback process")?;
            if !status.success() {
                anyhow::bail!("Playback process exited with status {status}");
            }
        }

        Ok(())
    }

    fn send_chunk(&mut self, chunk: &str) -> Result<()> {
        if chunk.trim().is_empty() {
            return Ok(());
        }

        let stdin = self
            .piper_stdin
            .as_mut()
            .context("Piper stdin is not available")?;
        stdin
            .write_all(chunk.as_bytes())
            .context("Failed writing text to piper")?;
        stdin
            .write_all(b"\n")
            .context("Failed writing newline to piper")?;
        stdin.flush().context("Failed flushing piper stdin")?;
        Ok(())
    }
}

fn config_path_for_model(model: &Path, explicit: Option<&Path>) -> PathBuf {
    if let Some(path) = explicit {
        return path.to_path_buf();
    }

    let mut s = model.as_os_str().to_string_lossy().to_string();
    s.push_str(".json");
    PathBuf::from(s)
}

fn read_sample_rate_from_config(path: &Path) -> Option<u32> {
    if !path.exists() {
        return None;
    }

    let txt = fs::read_to_string(path).ok()?;
    let value: serde_json::Value = serde_json::from_str(&txt).ok()?;
    value
        .get("audio")
        .and_then(|v| v.get("sample_rate"))
        .and_then(|v| v.as_u64())
        .map(|v| v as u32)
}

struct SentenceChunker {
    buffer: String,
    max_chars: usize,
}

impl SentenceChunker {
    fn new(max_chars: usize) -> Self {
        Self {
            buffer: String::new(),
            max_chars,
        }
    }

    fn push(&mut self, text: &str) -> Vec<String> {
        let mut out = Vec::new();
        for ch in text.chars() {
            self.buffer.push(ch);
            if is_boundary(ch) {
                let chunk = self.buffer.trim().to_string();
                if !chunk.is_empty() {
                    out.push(chunk);
                }
                self.buffer.clear();
                continue;
            }

            if self.buffer.len() >= self.max_chars && ch.is_whitespace() {
                let chunk = self.buffer.trim().to_string();
                if !chunk.is_empty() {
                    out.push(chunk);
                }
                self.buffer.clear();
            }
        }
        out
    }

    fn finish(&mut self) -> Option<String> {
        let tail = self.buffer.trim().to_string();
        self.buffer.clear();
        if tail.is_empty() {
            None
        } else {
            Some(tail)
        }
    }
}

fn is_boundary(ch: char) -> bool {
    matches!(ch, '.' | '!' | '?' | ';' | ':' | '\n')
}
