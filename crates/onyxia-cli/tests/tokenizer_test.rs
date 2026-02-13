//! Integration tests for tokenizer functionality.

use anyhow::Result;
use onyxia_cli::tokenizer::{ChatMessage, Tokenizer};
use std::path::PathBuf;
use std::env;

fn get_workspace_root() -> PathBuf {
    // CARGO_MANIFEST_DIR points to the crate dir (crates/onyxia-cli)
    // We need to go up two levels to get to workspace root
    let manifest_dir = env::var("CARGO_MANIFEST_DIR")
        .expect("CARGO_MANIFEST_DIR not set");
    PathBuf::from(manifest_dir)
        .parent()
        .expect("No parent directory")
        .parent()
        .expect("No workspace root")
        .to_path_buf()
}

fn get_tokenizer_path() -> PathBuf {
    get_workspace_root()
        .join("models")
        .join("gemma-3-270m-it-ONNX")
        .join("tokenizer.json")
}

fn get_chat_template_path() -> PathBuf {
    get_workspace_root()
        .join("models")
        .join("gemma-3-270m-it-ONNX")
        .join("chat_template.jinja")
}

#[test]
fn test_tokenizer_load() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?;
    
    // Gemma tokenizer has 262144 vocab entries (256k tokens)
    let vocab_size = tokenizer.vocab_size();
    assert_eq!(vocab_size, 262144, "Expected Gemma vocab size of 262144");
    
    Ok(())
}

#[test]
fn test_tokenizer_encode() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?;
    
    // Encode simple text without BOS
    let token_ids = tokenizer.encode("Hello, world!", false)?;
    assert!(!token_ids.is_empty(), "Token IDs should not be empty");
    assert!(token_ids.len() >= 2, "Expected at least 2 tokens for 'Hello, world!'");
    
    // Encode with BOS token
    let token_ids_with_bos = tokenizer.encode("Hello, world!", true)?;
    assert_eq!(token_ids_with_bos[0], 2, "First token should be BOS (2)");
    assert_eq!(token_ids_with_bos.len(), token_ids.len() + 1, "BOS should add one token");
    
    Ok(())
}

#[test]
fn test_tokenizer_decode() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?;
    
    // Encode then decode
    let original = "Hello, world!";
    let token_ids = tokenizer.encode(original, false)?;
    let decoded = tokenizer.decode(&token_ids, false)?;
    
    // The decoded text might have minor differences (spacing, etc.) but should contain the key words
    assert!(decoded.to_lowercase().contains("hello"), "Decoded text should contain 'hello'");
    assert!(decoded.to_lowercase().contains("world"), "Decoded text should contain 'world'");
    
    Ok(())
}

#[test]
fn test_tokenizer_roundtrip() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?;
    
    let test_strings = vec![
        "Hello!",
        "What is the capital of France?",
        "The quick brown fox jumps over the lazy dog.",
        "AI and machine learning are fascinating topics.",
    ];
    
    for original in test_strings {
        let token_ids = tokenizer.encode(original, false)?;
        let decoded = tokenizer.decode(&token_ids, false)?;
        
        // Normalize whitespace for comparison
        let original_normalized = original.trim().to_lowercase();
        let decoded_normalized = decoded.trim().to_lowercase();
        
        assert!(
            decoded_normalized.contains(&original_normalized) || 
            original_normalized.contains(&decoded_normalized),
            "Roundtrip failed: '{}' -> {:?} -> '{}'",
            original,
            token_ids,
            decoded
        );
    }
    
    Ok(())
}

#[test]
fn test_special_tokens() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?;
    
    assert_eq!(tokenizer.bos_token_id(), 2, "BOS token should be 2");
    assert_eq!(tokenizer.eos_token_id(), 1, "EOS token should be 1");
    assert_eq!(tokenizer.pad_token_id(), 0, "PAD token should be 0");
    
    assert!(tokenizer.is_eos(1), "Token 1 should be EOS");
    assert!(!tokenizer.is_eos(2), "Token 2 should not be EOS");
    assert!(!tokenizer.is_eos(100), "Token 100 should not be EOS");
    
    Ok(())
}

#[test]
fn test_chat_template_load() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?
        .with_chat_template_file(get_chat_template_path())?;
    
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello!".to_string(),
    }];
    
    let prompt = tokenizer.apply_chat_template(&messages, true)?;
    
    // Verify the template includes the expected format
    assert!(prompt.contains("<start_of_turn>user"), "Prompt should contain user turn marker");
    assert!(prompt.contains("Hello!"), "Prompt should contain the message content");
    assert!(prompt.contains("<end_of_turn>"), "Prompt should contain end turn marker");
    assert!(prompt.contains("<start_of_turn>model"), "Prompt should contain model turn marker for generation");
    
    Ok(())
}

#[test]
fn test_chat_template_multi_turn() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?
        .with_chat_template_file(get_chat_template_path())?;
    
    let messages = vec![
        ChatMessage {
            role: "user".to_string(),
            content: "What is 2+2?".to_string(),
        },
        ChatMessage {
            role: "assistant".to_string(),
            content: "The answer is 4.".to_string(),
        },
        ChatMessage {
            role: "user".to_string(),
            content: "What about 3+3?".to_string(),
        },
    ];
    
    let prompt = tokenizer.apply_chat_template(&messages, true)?;
    
    // Verify multi-turn structure
    assert!(prompt.contains("What is 2+2?"), "First user message should be present");
    assert!(prompt.contains("The answer is 4."), "Assistant response should be present");
    assert!(prompt.contains("What about 3+3?"), "Second user message should be present");
    
    // Assistant should be converted to model in the template
    assert!(prompt.contains("<start_of_turn>model"), "Assistant role should be converted to model");
    
    Ok(())
}

#[test]
fn test_chat_template_without_generation_prompt() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?
        .with_chat_template_file(get_chat_template_path())?;
    
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "Hello!".to_string(),
    }];
    
    let prompt_with_gen = tokenizer.apply_chat_template(&messages, true)?;
    let prompt_without_gen = tokenizer.apply_chat_template(&messages, false)?;
    
    // With generation prompt should have model turn at the end
    assert!(prompt_with_gen.ends_with("<start_of_turn>model\n"), 
        "Prompt with generation should end with model turn");
    
    // Without generation prompt should not have trailing model turn
    assert!(!prompt_without_gen.ends_with("<start_of_turn>model\n"),
        "Prompt without generation should not end with model turn");
    
    Ok(())
}

#[test]
fn test_encode_chat_template_output() -> Result<()> {
    let tokenizer = Tokenizer::from_file(get_tokenizer_path())?
        .with_chat_template_file(get_chat_template_path())?;
    
    let messages = vec![ChatMessage {
        role: "user".to_string(),
        content: "What is the capital of France?".to_string(),
    }];
    
    // Apply chat template
    let prompt = tokenizer.apply_chat_template(&messages, true)?;
    
    // Encode the formatted prompt
    let token_ids = tokenizer.encode(&prompt, true)?;
    
    // Verify we got a reasonable number of tokens
    assert!(!token_ids.is_empty(), "Token IDs should not be empty");
    assert!(token_ids.len() > 5, "Expected more than 5 tokens for formatted prompt");
    
    // First token should be BOS
    assert_eq!(token_ids[0], 2, "First token should be BOS");
    
    Ok(())
}
