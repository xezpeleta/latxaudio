use serde::Serialize;

#[derive(Clone, Debug, Serialize)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

pub struct Conversation {
    system_prompt: Option<String>,
    history: Vec<(String, String)>,
    max_turns: usize,
}

impl Conversation {
    pub fn new(system_prompt: Option<String>, max_turns: usize) -> Self {
        Self {
            system_prompt,
            history: Vec::new(),
            max_turns,
        }
    }

    pub fn build_messages(&self, user_text: &str) -> Vec<ChatMessage> {
        let mut out = Vec::new();

        if let Some(prompt) = &self.system_prompt {
            out.push(ChatMessage {
                role: "system".to_string(),
                content: prompt.clone(),
            });
        }

        for (user, assistant) in &self.history {
            out.push(ChatMessage {
                role: "user".to_string(),
                content: user.clone(),
            });
            out.push(ChatMessage {
                role: "assistant".to_string(),
                content: assistant.clone(),
            });
        }

        out.push(ChatMessage {
            role: "user".to_string(),
            content: user_text.to_string(),
        });

        out
    }

    pub fn add_turn(&mut self, user_text: String, assistant_text: String) {
        self.history.push((user_text, assistant_text));
        while self.history.len() > self.max_turns {
            self.history.remove(0);
        }
    }
}
