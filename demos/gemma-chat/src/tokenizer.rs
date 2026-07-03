//! Tokenization support for LLM inference.
//!
//! Wraps the `tokenizers` crate to handle encoding text to token IDs and
//! decoding token IDs back to text. Includes chat template rendering using
//! minijinja for instruction-tuned models like Gemma.

use anyhow::{Context, Result};
use minijinja::{Environment, context};
use std::path::Path;
use tokenizers::Tokenizer as HfTokenizer;

/// Special token IDs for Gemma models.
#[derive(Debug, Clone)]
pub struct SpecialTokens {
    pub bos: u32,
    pub eos: u32,
    /// Gemma's turn terminator `<end_of_turn>`. Instruction-tuned chat models
    /// end each turn with this token rather than `<eos>`, so generation must
    /// stop on it. `None` for base models that don't define it.
    pub end_of_turn: Option<u32>,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: 2, // <bos> token
            eos: 1, // <eos> token
            end_of_turn: None,
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// High-level tokenizer: encodes text to token IDs, decodes IDs back to
/// text, and renders chat templates for instruction-tuned models. Wraps
/// HuggingFace's `tokenizers` crate.
pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: SpecialTokens,
    chat_template: Option<String>,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    #[allow(dead_code)] // native-only; wasm fetches bytes and uses `from_bytes`
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        // Discover Gemma's turn terminator so chat generation stops on it.
        let end_of_turn = inner.token_to_id("<end_of_turn>");

        Ok(Self {
            inner,
            special_tokens: SpecialTokens {
                end_of_turn,
                ..SpecialTokens::default()
            },
            chat_template: None,
        })
    }

    /// Load a tokenizer from an in-memory `tokenizer.json` byte buffer.
    ///
    /// Used on the web, where the file is fetched over HTTP rather than read
    /// from disk.
    #[allow(dead_code)] // used on wasm; native uses `from_file`
    pub fn from_bytes(bytes: &[u8]) -> Result<Self> {
        let inner = HfTokenizer::from_bytes(bytes)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        let end_of_turn = inner.token_to_id("<end_of_turn>");

        Ok(Self {
            inner,
            special_tokens: SpecialTokens {
                end_of_turn,
                ..SpecialTokens::default()
            },
            chat_template: None,
        })
    }

    /// Load a chat template from a Jinja file.
    #[allow(dead_code)] // native-only; wasm fetches the template and uses `with_chat_template`
    pub fn with_chat_template_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let template =
            std::fs::read_to_string(path).context("Failed to read chat template file")?;
        self.chat_template = Some(template);
        Ok(self)
    }

    /// Set a chat template string directly.
    #[allow(dead_code)] // used on wasm (template arrives over HTTP); native reads the file
    pub fn with_chat_template(mut self, template: String) -> Self {
        self.chat_template = Some(template);
        self
    }

    /// Encode text to a vector of token IDs (as i64 for ONNX models).
    /// With `add_bos`, a BOS (beginning-of-sequence) token is prepended.
    pub fn encode(&self, text: &str, add_bos: bool) -> Result<Vec<i64>> {
        let encoding = self
            .inner
            .encode(text, false)
            .map_err(|e| anyhow::anyhow!("Encoding failed: {}", e))?;

        let mut token_ids: Vec<i64> = encoding.get_ids().iter().map(|&id| id as i64).collect();

        if add_bos {
            token_ids.insert(0, self.special_tokens.bos as i64);
        }

        Ok(token_ids)
    }

    /// Decode a sequence of token IDs back to text. With
    /// `skip_special_tokens`, special tokens (BOS, EOS, PAD) are removed
    /// from the output.
    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        // Convert i64 back to u32 for the tokenizer
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

        let text = self
            .inner
            .decode(&ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(text)
    }

    /// Check if a token ID is the EOS (end-of-sequence) token.
    pub fn is_eos(&self, token_id: i64) -> bool {
        token_id == self.special_tokens.eos as i64
            || self.special_tokens.end_of_turn == Some(token_id as u32)
    }

    /// Render the conversation through the Jinja chat template (via
    /// minijinja). With `add_generation_prompt`, the model-turn prompt is
    /// appended so the model continues as the assistant.
    pub fn apply_chat_template(
        &self,
        messages: &[ChatMessage],
        add_generation_prompt: bool,
    ) -> Result<String> {
        let template = self.chat_template.as_ref().context(
            "No chat template loaded. Use with_chat_template_file() or with_chat_template()",
        )?;

        let mut env = Environment::new();
        env.add_template("chat", template)
            .context("Failed to add chat template to minijinja environment")?;

        let tmpl = env
            .get_template("chat")
            .context("Failed to get chat template")?;

        // Convert messages to minijinja-compatible format
        let messages_data: Vec<_> = messages
            .iter()
            .map(|msg| {
                context! {
                    role => &msg.role,
                    content => &msg.content,
                }
            })
            .collect();

        let rendered = tmpl
            .render(context! {
                messages => messages_data,
                add_generation_prompt => add_generation_prompt,
                bos_token => "<bos>",
            })
            .context("Failed to render chat template")?;

        Ok(rendered)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_special_tokens_default() {
        let tokens = SpecialTokens::default();
        assert_eq!(tokens.bos, 2);
        assert_eq!(tokens.eos, 1);
    }

    #[test]
    #[ignore = "needs models/gemma-3-270m-it-ONNX (not in git); run via `just test-all`"]
    fn test_tokenizer_special_token_checks() {
        // Get workspace root by going up from CARGO_MANIFEST_DIR
        let manifest_dir = std::env::var("CARGO_MANIFEST_DIR").expect("CARGO_MANIFEST_DIR not set");
        let workspace_root = std::path::PathBuf::from(manifest_dir)
            .parent()
            .expect("No parent directory")
            .parent()
            .expect("No workspace root")
            .to_path_buf();

        let tokenizer_path = workspace_root
            .join("models")
            .join("gemma-3-270m-it-ONNX")
            .join("tokenizer.json");

        let tokenizer_data = SpecialTokens::default();
        let tokenizer = Tokenizer {
            inner: HfTokenizer::from_file(tokenizer_path).expect("Failed to load tokenizer"),
            special_tokens: tokenizer_data,
            chat_template: None,
        };

        assert!(tokenizer.is_eos(1));
        assert!(!tokenizer.is_eos(2));
    }

    #[test]
    fn test_chat_message() {
        let msg = ChatMessage {
            role: "user".to_string(),
            content: "Hello!".to_string(),
        };
        assert_eq!(msg.role, "user");
        assert_eq!(msg.content, "Hello!");
    }
}
