use crate::audio::{TARGET_SAMPLE_RATE, VAD_WINDOW_SIZE};
use anyhow::{Context, Result};
use sherpa_onnx::{
    DisplayManager, OfflineRecognizer, OfflineRecognizerConfig, OfflineTransducerModelConfig,
    SileroVadModelConfig, VadModelConfig, VoiceActivityDetector,
};
use std::path::PathBuf;
use std::time::{Duration, Instant};

const PARTIAL_DECODE_INTERVAL: Duration = Duration::from_millis(200);

pub struct SttConfig {
    pub model_dir: PathBuf,
    pub vad_model: PathBuf,
    pub provider: String,
    pub num_threads: i32,
    pub vad_threshold: f32,
    pub vad_min_silence: f32,
    pub vad_min_speech: f32,
    pub max_speech_duration: f32,
    pub debug: bool,
}

pub struct SttEngine {
    vad: VoiceActivityDetector,
    recognizer: OfflineRecognizer,
    display: DisplayManager,
    buffer: Vec<f32>,
    offset: usize,
    speech_started: bool,
    partial_timer: Instant,
}

impl SttEngine {
    pub fn new(cfg: &SttConfig) -> Result<Self> {
        let encoder = cfg.model_dir.join("encoder.int8.onnx");
        let decoder = cfg.model_dir.join("decoder.int8.onnx");
        let joiner = cfg.model_dir.join("joiner.int8.onnx");
        let tokens = cfg.model_dir.join("tokens.txt");

        for (label, path) in [
            ("encoder", &encoder),
            ("decoder", &decoder),
            ("joiner", &joiner),
            ("tokens", &tokens),
            ("vad_model", &cfg.vad_model),
        ] {
            if !path.exists() {
                anyhow::bail!(
                    "Missing {label}: {}\nRun ./scripts/download-models.sh to download STT models.",
                    path.display()
                );
            }
        }

        let mut silero = SileroVadModelConfig::default();
        silero.model = Some(cfg.vad_model.to_string_lossy().into_owned());
        silero.threshold = cfg.vad_threshold;
        silero.min_silence_duration = cfg.vad_min_silence;
        silero.min_speech_duration = cfg.vad_min_speech;
        silero.max_speech_duration = cfg.max_speech_duration;

        let vad_cfg = VadModelConfig {
            silero_vad: silero,
            ten_vad: Default::default(),
            sample_rate: TARGET_SAMPLE_RATE as i32,
            num_threads: 1,
            provider: Some("cpu".to_string()),
            debug: cfg.debug,
        };

        println!("[latxaudio] Loading VAD model...");
        let vad = VoiceActivityDetector::create(&vad_cfg, 60.0)
            .context("Failed to create VoiceActivityDetector")?;

        let mut asr_cfg = OfflineRecognizerConfig::default();
        asr_cfg.model_config.transducer = OfflineTransducerModelConfig {
            encoder: Some(encoder.to_string_lossy().into_owned()),
            decoder: Some(decoder.to_string_lossy().into_owned()),
            joiner: Some(joiner.to_string_lossy().into_owned()),
        };
        asr_cfg.model_config.tokens = Some(tokens.to_string_lossy().into_owned());
        asr_cfg.model_config.provider = Some(cfg.provider.clone());
        asr_cfg.model_config.num_threads = cfg.num_threads;
        asr_cfg.model_config.debug = cfg.debug;
        asr_cfg.decoding_method = Some("greedy_search".to_string());

        println!(
            "[latxaudio] Loading ASR model ({})...",
            cfg.provider.to_uppercase()
        );
        let recognizer =
            OfflineRecognizer::create(&asr_cfg).context("Failed to create OfflineRecognizer")?;

        Ok(Self {
            vad,
            recognizer,
            display: DisplayManager::new(),
            buffer: Vec::new(),
            offset: 0,
            speech_started: false,
            partial_timer: Instant::now(),
        })
    }

    pub fn accept_audio(&mut self, chunk_16k_mono: &[f32]) -> Option<String> {
        self.buffer.extend_from_slice(chunk_16k_mono);

        while self.offset + VAD_WINDOW_SIZE <= self.buffer.len() {
            self.vad
                .accept_waveform(&self.buffer[self.offset..self.offset + VAD_WINDOW_SIZE]);

            if !self.speech_started && self.vad.detected() {
                self.speech_started = true;
                self.partial_timer = Instant::now();
            }

            self.offset += VAD_WINDOW_SIZE;
        }

        if !self.speech_started && self.buffer.len() > 10 * VAD_WINDOW_SIZE {
            let drop_n = self.buffer.len() - 10 * VAD_WINDOW_SIZE;
            self.buffer.drain(..drop_n);
            self.offset = self.offset.saturating_sub(drop_n);
        }

        if self.speech_started && self.partial_timer.elapsed() >= PARTIAL_DECODE_INTERVAL {
            let asr_stream = self.recognizer.create_stream();
            asr_stream.accept_waveform(TARGET_SAMPLE_RATE as i32, &self.buffer);
            self.recognizer.decode(&asr_stream);

            if let Some(result) = asr_stream.get_result() {
                if !result.text.is_empty() {
                    self.display.update_text(&result.text);
                    self.display.render();
                }
            }

            self.partial_timer = Instant::now();
        }

        let mut processed_segments = false;
        let mut best_text: Option<String> = None;
        let mut best_score: usize = 0;
        while let Some(segment) = self.vad.front() {
            self.vad.pop();
            processed_segments = true;

            if let Some(text) = self.decode_segment_text(segment.samples()) {
                let score = transcript_score(&text);
                if score >= best_score {
                    best_score = score;
                    best_text = Some(text);
                }
            }
        }

        if processed_segments {
            self.buffer.clear();
            self.offset = 0;
            self.speech_started = false;
        }

        if let Some(final_text) = best_text {
            self.display.update_text(&final_text);
            self.display.finalize_sentence();
            self.display.render();

            return Some(final_text);
        }

        None
    }

    pub fn reset(&mut self) {
        self.buffer.clear();
        self.offset = 0;
        self.speech_started = false;
        self.partial_timer = Instant::now();
        while self.vad.front().is_some() {
            self.vad.pop();
        }
    }

    fn decode_segment_text(&self, samples: &[f32]) -> Option<String> {
        let asr_stream = self.recognizer.create_stream();
        asr_stream.accept_waveform(TARGET_SAMPLE_RATE as i32, samples);
        self.recognizer.decode(&asr_stream);

        let result = asr_stream.get_result()?;
        let text = result.text.trim();
        if text.is_empty() {
            None
        } else {
            Some(text.to_string())
        }
    }
}

fn transcript_score(text: &str) -> usize {
    let words = text.split_whitespace().count();
    words * 16 + text.chars().count()
}
