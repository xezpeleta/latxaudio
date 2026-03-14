use crate::chat::ChatMessage;
use anyhow::{Context, Result};
use reqwest::blocking::Client;
use serde_json::{json, Value};
use std::io::{BufRead, BufReader};
use std::time::Duration;

pub struct LlmClient {
    client: Client,
    base_url: String,
    api_key: Option<String>,
    model: String,
    temperature: f32,
}

impl LlmClient {
    pub fn new(
        base_url: String,
        api_key: Option<String>,
        model: String,
        temperature: f32,
        timeout_secs: u64,
    ) -> Result<Self> {
        let client = Client::builder()
            .timeout(Duration::from_secs(timeout_secs))
            .build()
            .context("Failed to create HTTP client")?;

        Ok(Self {
            client,
            base_url,
            api_key,
            model,
            temperature,
        })
    }

    pub fn stream_chat<F>(&self, messages: &[ChatMessage], mut on_delta: F) -> Result<String>
    where
        F: FnMut(&str) -> Result<()>,
    {
        let url = format!("{}/chat/completions", self.base_url.trim_end_matches('/'));
        let body = json!({
            "model": self.model,
            "stream": true,
            "temperature": self.temperature,
            "messages": messages,
        });

        let mut req = self
            .client
            .post(url)
            .header("Content-Type", "application/json");

        if let Some(api_key) = &self.api_key {
            if !api_key.trim().is_empty() {
                req = req.header("Authorization", format!("Bearer {api_key}"));
            }
        }

        let resp = req
            .json(&body)
            .send()
            .context("Failed to send LLM request")?
            .error_for_status()
            .context("LLM request failed")?;

        let mut reader = BufReader::new(resp);
        let mut line = String::new();
        let mut full = String::new();

        loop {
            line.clear();
            let n = reader
                .read_line(&mut line)
                .context("Failed to read LLM stream")?;
            if n == 0 {
                break;
            }

            let trimmed = line.trim();
            if !trimmed.starts_with("data:") {
                continue;
            }

            let payload = trimmed.trim_start_matches("data:").trim();
            if payload == "[DONE]" {
                break;
            }

            let Ok(chunk) = serde_json::from_str::<Value>(payload) else {
                continue;
            };

            if let Some(delta) = extract_delta_text(&chunk) {
                if !delta.is_empty() {
                    on_delta(delta)?;
                    full.push_str(delta);
                }
            }
        }

        Ok(full)
    }
}

fn extract_delta_text(chunk: &Value) -> Option<&str> {
    let choices = chunk.get("choices")?.as_array()?;
    let first = choices.first()?;
    let delta = first.get("delta")?;

    if let Some(s) = delta.get("content").and_then(|v| v.as_str()) {
        return Some(s);
    }

    if let Some(parts) = delta.get("content").and_then(|v| v.as_array()) {
        for item in parts {
            if item.get("type").and_then(|v| v.as_str()) == Some("text") {
                if let Some(text) = item.get("text").and_then(|v| v.as_str()) {
                    return Some(text);
                }
            }
        }
    }

    None
}
