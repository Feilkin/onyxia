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
    pub pad: u32,
}

impl Default for SpecialTokens {
    fn default() -> Self {
        Self {
            bos: 2, // <bos> token
            eos: 1, // <eos> token
            pad: 0, // <pad> token
        }
    }
}

/// A single message in a chat conversation.
#[derive(Debug, Clone)]
pub struct ChatMessage {
    pub role: String,
    pub content: String,
}

/// High-level tokenizer for LLM text processing.
///
/// Wraps HuggingFace's `tokenizers` crate and provides methods for:
/// - Encoding text to token IDs (for model input)
/// - Decoding token IDs back to text (for generation output)
/// - Formatting chat templates for instruction-tuned models
pub struct Tokenizer {
    inner: HfTokenizer,
    special_tokens: SpecialTokens,
    chat_template: Option<String>,
}

impl Tokenizer {
    /// Load a tokenizer from a `tokenizer.json` file.
    ///
    /// # Example
    /// ```no_run
    /// use onyxia_cli::tokenizer::Tokenizer;
    /// let tokenizer = Tokenizer::from_file("models/gemma-3-270m-it-ONNX/tokenizer.json")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn from_file<P: AsRef<Path>>(path: P) -> Result<Self> {
        let inner = HfTokenizer::from_file(path)
            .map_err(|e| anyhow::anyhow!("Failed to load tokenizer: {}", e))?;

        Ok(Self {
            inner,
            special_tokens: SpecialTokens::default(),
            chat_template: None,
        })
    }

    /// Load a chat template from a Jinja file.
    ///
    /// # Example
    /// ```no_run
    /// use onyxia_cli::tokenizer::Tokenizer;
    /// let tokenizer = Tokenizer::from_file("models/gemma-3-270m-it-ONNX/tokenizer.json")?
    ///     .with_chat_template_file("models/gemma-3-270m-it-ONNX/chat_template.jinja")?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn with_chat_template_file<P: AsRef<Path>>(mut self, path: P) -> Result<Self> {
        let template =
            std::fs::read_to_string(path).context("Failed to read chat template file")?;
        self.chat_template = Some(template);
        Ok(self)
    }

    /// Set a chat template string directly.
    pub fn with_chat_template(mut self, template: String) -> Self {
        self.chat_template = Some(template);
        self
    }

    /// Set custom special token IDs.
    pub fn with_special_tokens(mut self, special_tokens: SpecialTokens) -> Self {
        self.special_tokens = special_tokens;
        self
    }

    /// Encode text to a vector of token IDs (as i64 for ONNX models).
    ///
    /// Optionally adds a BOS (beginning-of-sequence) token at the start.
    ///
    /// # Example
    /// ```no_run
    /// # use onyxia_cli::tokenizer::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    /// let token_ids = tokenizer.encode("Hello, world!", true)?;
    /// assert!(token_ids[0] == 2); // BOS token
    /// # Ok::<(), anyhow::Error>(())
    /// ```
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

    /// Decode a sequence of token IDs back to text.
    ///
    /// If `skip_special_tokens` is true, special tokens (BOS, EOS, PAD) are
    /// removed from the output.
    ///
    /// # Example
    /// ```no_run
    /// # use onyxia_cli::tokenizer::Tokenizer;
    /// # let tokenizer = Tokenizer::from_file("tokenizer.json")?;
    /// let text = tokenizer.decode(&[2, 4521, 235269, 2134, 235341], true)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
    pub fn decode(&self, token_ids: &[i64], skip_special_tokens: bool) -> Result<String> {
        // Convert i64 back to u32 for the tokenizer
        let ids: Vec<u32> = token_ids.iter().map(|&id| id as u32).collect();

        let text = self
            .inner
            .decode(&ids, skip_special_tokens)
            .map_err(|e| anyhow::anyhow!("Decoding failed: {}", e))?;

        Ok(text)
    }

    /// Get the vocabulary size.
    pub fn vocab_size(&self) -> usize {
        self.inner.get_vocab_size(false)
    }

    /// Check if a token ID is the EOS (end-of-sequence) token.
    pub fn is_eos(&self, token_id: i64) -> bool {
        token_id == self.special_tokens.eos as i64
    }

    /// Get the EOS token ID.
    pub fn eos_token_id(&self) -> i64 {
        self.special_tokens.eos as i64
    }

    /// Get the BOS token ID.
    pub fn bos_token_id(&self) -> i64 {
        self.special_tokens.bos as i64
    }

    /// Get the PAD token ID.
    pub fn pad_token_id(&self) -> i64 {
        self.special_tokens.pad as i64
    }

    /// Apply the chat template to format a conversation.
    ///
    /// Uses minijinja to render the Jinja template with the provided messages.
    ///
    /// # Arguments
    /// - `messages`: List of chat messages with role and content
    /// - `add_generation_prompt`: Whether to add the model turn prompt at the end
    ///
    /// # Example
    /// ```no_run
    /// use onyxia_cli::tokenizer::{Tokenizer, ChatMessage};
    /// # let tokenizer = Tokenizer::from_file("tokenizer.json")?
    /// #     .with_chat_template_file("chat_template.jinja")?;
    /// let messages = vec![
    ///     ChatMessage { role: "user".to_string(), content: "Hello!".to_string() }
    /// ];
    /// let prompt = tokenizer.apply_chat_template(&messages, true)?;
    /// # Ok::<(), anyhow::Error>(())
    /// ```
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

/// Format a user message with the Gemma instruction chat template (simple version).
///
/// Gemma uses the format:
/// ```text
/// <start_of_turn>user
/// {user_message}<end_of_turn>
/// <start_of_turn>model
/// ```
///
/// This is a simple fallback if you don't want to use the full Jinja template.
///
/// # Example
/// ```
/// use onyxia_cli::tokenizer::format_chat_prompt;
/// let prompt = format_chat_prompt("What is the capital of France?");
/// assert!(prompt.contains("<start_of_turn>user"));
/// assert!(prompt.contains("<start_of_turn>model"));
/// ```
pub fn format_chat_prompt(user_message: &str) -> String {
    format!(
        "<start_of_turn>user\n{}<end_of_turn>\n<start_of_turn>model\n",
        user_message
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_format_chat_prompt() {
        let prompt = format_chat_prompt("Hello!");
        assert!(prompt.contains("<start_of_turn>user"));
        assert!(prompt.contains("Hello!"));
        assert!(prompt.contains("<end_of_turn>"));
        assert!(prompt.contains("<start_of_turn>model"));
    }

    #[test]
    fn test_special_tokens_default() {
        let tokens = SpecialTokens::default();
        assert_eq!(tokens.bos, 2);
        assert_eq!(tokens.eos, 1);
        assert_eq!(tokens.pad, 0);
    }

    #[test]
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
        assert_eq!(tokenizer.eos_token_id(), 1);
        assert_eq!(tokenizer.bos_token_id(), 2);
        assert_eq!(tokenizer.pad_token_id(), 0);
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
