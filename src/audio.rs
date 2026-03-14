use anyhow::{Context, Result};
use cpal::traits::{DeviceTrait, HostTrait};
use cpal::{SampleFormat, SampleRate, StreamConfig};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::Arc;

pub const TARGET_SAMPLE_RATE: u32 = 16_000;
pub const VAD_WINDOW_SIZE: usize = 512;

pub fn select_input_device(host: &cpal::Host, index: Option<usize>) -> Result<cpal::Device> {
    let devices: Vec<cpal::Device> = host
        .input_devices()
        .context("Failed to enumerate input devices")?
        .collect();

    if devices.is_empty() {
        anyhow::bail!(
            "No input devices found. On Linux, ensure ALSA/PipeWire is available and your user is in the 'audio' group."
        );
    }

    println!("Available input devices:");
    let default_name = host
        .default_input_device()
        .and_then(|d| d.name().ok())
        .unwrap_or_default();

    for (i, dev) in devices.iter().enumerate() {
        let name = dev.name().unwrap_or_else(|_| "<unknown>".to_string());
        let marker = if name == default_name { "*" } else { " " };
        println!("  [{}]{} {}", i, marker, name);
    }
    println!();

    match index {
        Some(i) => devices
            .into_iter()
            .nth(i)
            .with_context(|| format!("No device at index {i}")),
        None => host
            .default_input_device()
            .context("No default input device found"),
    }
}

pub fn build_audio_stream(
    device: &cpal::Device,
    tx: mpsc::SyncSender<Vec<f32>>,
    capture_enabled: Arc<AtomicBool>,
) -> Result<(cpal::Stream, u32)> {
    let config_16k = StreamConfig {
        channels: 1,
        sample_rate: SampleRate(TARGET_SAMPLE_RATE),
        buffer_size: cpal::BufferSize::Default,
    };

    let tx_clone = tx.clone();
    let enabled_clone = Arc::clone(&capture_enabled);
    match device.build_input_stream(
        &config_16k,
        move |data: &[f32], _: &_| {
            if !enabled_clone.load(Ordering::Acquire) {
                return;
            }
            let _ = tx_clone.try_send(data.to_vec());
        },
        |err| eprintln!("[audio] stream error: {err}"),
        None,
    ) {
        Ok(stream) => return Ok((stream, TARGET_SAMPLE_RATE)),
        Err(e) => {
            eprintln!("[audio] 16 kHz mono not supported ({e}); using device default");
        }
    }

    let default_cfg = device
        .default_input_config()
        .context("Failed to get default input config")?;

    let channels = default_cfg.channels() as usize;
    let actual_rate = default_cfg.sample_rate().0;
    let stream_config = default_cfg.config();
    let err_fn = |err| eprintln!("[audio] stream error: {err}");

    let stream = match default_cfg.sample_format() {
        SampleFormat::F32 => {
            let enabled = Arc::clone(&capture_enabled);
            device.build_input_stream(
                &stream_config,
                move |data: &[f32], _: &_| {
                    if !enabled.load(Ordering::Acquire) {
                        return;
                    }
                    let mono = mix_to_mono_f32(data, channels);
                    let _ = tx.try_send(mono);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::I16 => {
            let enabled = Arc::clone(&capture_enabled);
            device.build_input_stream(
                &stream_config,
                move |data: &[i16], _: &_| {
                    if !enabled.load(Ordering::Acquire) {
                        return;
                    }
                    let mono = mix_to_mono_i16(data, channels);
                    let _ = tx.try_send(mono);
                },
                err_fn,
                None,
            )?
        }
        SampleFormat::U16 => {
            let enabled = Arc::clone(&capture_enabled);
            device.build_input_stream(
                &stream_config,
                move |data: &[u16], _: &_| {
                    if !enabled.load(Ordering::Acquire) {
                        return;
                    }
                    let mono = mix_to_mono_u16(data, channels);
                    let _ = tx.try_send(mono);
                },
                err_fn,
                None,
            )?
        }
        fmt => anyhow::bail!("Unsupported sample format: {fmt:?}"),
    };

    eprintln!("[audio] capturing at {actual_rate} Hz, {channels} ch - will resample to 16 kHz");
    Ok((stream, actual_rate))
}

pub fn resample(input: &[f32], from: u32, to: u32) -> Vec<f32> {
    if from == to || input.is_empty() {
        return input.to_vec();
    }

    let ratio = from as f64 / to as f64;
    let out_len = ((input.len() as f64) / ratio).ceil() as usize;
    let mut out = Vec::with_capacity(out_len);

    for i in 0..out_len {
        let pos = i as f64 * ratio;
        let lo = pos as usize;
        let frac = (pos - lo as f64) as f32;
        let a = input.get(lo).copied().unwrap_or(0.0);
        let b = input.get(lo + 1).copied().unwrap_or(0.0);
        out.push(a + frac * (b - a));
    }

    out
}

fn mix_to_mono_f32(data: &[f32], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .map(|f| f.iter().copied().sum::<f32>() / channels as f32)
        .collect()
}

fn mix_to_mono_i16(data: &[i16], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .map(|f| f.iter().map(|&s| s as f32 / i16::MAX as f32).sum::<f32>() / channels as f32)
        .collect()
}

fn mix_to_mono_u16(data: &[u16], channels: usize) -> Vec<f32> {
    data.chunks(channels)
        .map(|f| {
            f.iter()
                .map(|&s| (s as f32 - 32768.0) / 32768.0)
                .sum::<f32>()
                / channels as f32
        })
        .collect()
}
