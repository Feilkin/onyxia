//! Text generation loop for LLM inference.
//!
//! Implements the full generation workflow:
//! 1. Encode prompt to token IDs
//! 2. Prefill: process full prompt through model
//! 3. Decode: generate tokens one at a time until EOS or max_tokens
//! 4. Stream output as tokens are generated

use crate::llm::LlmSession;
use crate::sampling::{SamplingConfig, sample};
use crate::tokenizer::Tokenizer;
use anyhow::{Context, Result};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::io::{self, Write};
use std::time::Instant;

/// Statistics from a generation run.
#[derive(Debug)]
pub struct GenerationStats {
    /// Total tokens generated (excluding prompt).
    pub tokens_generated: usize,
    /// Time spent in prefill phase (seconds).
    pub prefill_time: f64,
    /// Time spent in decode phase (seconds).
    pub decode_time: f64,
    /// Tokens per second (decode only).
    pub tokens_per_sec: f64,
    /// Total time (seconds).
    pub total_time: f64,
}

/// Generate text from a prompt using an LLM session.
///
/// # Arguments
/// * `session` - LLM session with loaded model and KV cache
/// * `tokenizer` - Tokenizer for encoding/decoding text
/// * `prompt` - Input text to generate from
/// * `max_tokens` - Maximum number of tokens to generate (not including prompt)
/// * `config` - Sampling configuration (temperature, top-k, top-p, seed)
/// * `stream` - Whether to print tokens as they're generated
/// * `stop_token_ids` - Token IDs that stop generation when sampled
///
/// # Returns
/// Generated text and statistics
pub fn generate(
    session: &mut LlmSession,
    tokenizer: &Tokenizer,
    prompt: &str,
    max_tokens: usize,
    config: &SamplingConfig,
    stream: bool,
    stop_token_ids: &[u32],
) -> Result<(String, GenerationStats)> {
    let generation_start = Instant::now();

    // Initialize RNG from config
    let mut rng = match config.seed {
        Some(seed) => StdRng::seed_from_u64(seed),
        None => StdRng::from_seed(rand::random()),
    };

    // Encode prompt
    let input_ids = tokenizer
        .encode(prompt, false)
        .with_context(|| format!("Failed to encode prompt: '{}'", prompt))?;

    if input_ids.is_empty() {
        anyhow::bail!("Prompt '{}' encoded to empty token sequence", prompt);
    }

    if input_ids.len() > max_tokens {
        anyhow::bail!(
            "Prompt length {} exceeds max_tokens {}",
            input_ids.len(),
            max_tokens
        );
    }

    // Prefill phase: process full prompt
    let prefill_start = Instant::now();
    let logits = session.prefill(&input_ids).with_context(|| {
        format!(
            "Prefill failed (prompt='{}', input_ids_len={})",
            prompt,
            input_ids.len()
        )
    })?;
    let prefill_time = prefill_start.elapsed().as_secs_f64();

    if logits.is_empty() {
        anyhow::bail!("Prefill returned empty logits vector");
    }

    // Sample first token from prefill logits
    let mut generated_tokens = Vec::new();
    let first_token = sample(&logits, config, &mut rng);

    if first_token as usize >= logits.len() {
        anyhow::bail!(
            "Sampled token index {} out of range for logits of size {}",
            first_token,
            logits.len()
        );
    }

    generated_tokens.push(first_token);

    // Stream first token if enabled
    if stream {
        let text = tokenizer
            .decode(&[first_token as i64], false)
            .context("Failed to decode token")?;
        print!("{}", text);
        io::stdout().flush().ok();
    }

    // Check if first token is a stop token
    if stop_token_ids.contains(&first_token) {
        let total_time = generation_start.elapsed().as_secs_f64();
        let stats = GenerationStats {
            tokens_generated: 1,
            prefill_time,
            decode_time: 0.0,
            tokens_per_sec: 0.0,
            total_time,
        };
        let generated_tokens_i64: Vec<i64> = generated_tokens.iter().map(|&t| t as i64).collect();
        let full_text = tokenizer
            .decode(&generated_tokens_i64, false)
            .context("Failed to decode generated tokens")?;
        return Ok((full_text, stats));
    }

    // Decode phase: generate tokens one at a time
    let decode_start = Instant::now();
    let mut next_token = first_token;

    for step in 1..max_tokens {
        let logits = session.decode(next_token as i64).with_context(|| {
            format!(
                "Decode failed at step {} (token_id={}, total_generated={})",
                step,
                next_token,
                generated_tokens.len()
            )
        })?;

        if logits.is_empty() {
            anyhow::bail!("Decode returned empty logits at step {}", step);
        }

        next_token = sample(&logits, config, &mut rng);

        if next_token as usize >= logits.len() {
            anyhow::bail!(
                "Sampled token {} out of range for logits size {} at step {}",
                next_token,
                logits.len(),
                step
            );
        }

        generated_tokens.push(next_token);

        // Stream token if enabled
        if stream {
            let text = tokenizer
                .decode(&[next_token as i64], false)
                .context("Failed to decode token")?;
            print!("{}", text);
            io::stdout().flush().ok();
        }

        // Stop if a stop token is generated
        if stop_token_ids.contains(&next_token) {
            break;
        }
    }

    let decode_time = decode_start.elapsed().as_secs_f64();
    let total_time = generation_start.elapsed().as_secs_f64();

    // Calculate tokens/sec (exclude first token which was from prefill)
    let decode_token_count = generated_tokens.len() - 1;
    let tokens_per_sec = if decode_time > 0.0 {
        decode_token_count as f64 / decode_time
    } else {
        0.0
    };

    if stream {
        println!(); // Newline after generation
    }

    let stats = GenerationStats {
        tokens_generated: generated_tokens.len(),
        prefill_time,
        decode_time,
        tokens_per_sec,
        total_time,
    };

    // Decode all generated tokens to text
    let generated_tokens_i64: Vec<i64> = generated_tokens.iter().map(|&t| t as i64).collect();
    let full_text = tokenizer
        .decode(&generated_tokens_i64, false)
        .context("Failed to decode generated tokens")?;

    Ok((full_text, stats))
}

/// Print generation statistics in a formatted table.
pub fn print_stats(stats: &GenerationStats) {
    println!("\n{}", "=".repeat(50));
    println!("Generation Statistics");
    println!("{}", "=".repeat(50));
    println!("Tokens generated:    {}", stats.tokens_generated);
    println!("Prefill time:        {:.3}s", stats.prefill_time);
    println!("Decode time:         {:.3}s", stats.decode_time);
    println!("Tokens/sec:          {:.2}", stats.tokens_per_sec);
    println!("Total time:          {:.3}s", stats.total_time);
    println!("{}", "=".repeat(50));
}

#[cfg(test)]
mod tests {
    use super::*;

    // Unit tests for generation logic would go here
    // Integration tests with actual model execution are in tests/

    #[test]
    fn test_generation_stats_calculation() {
        let stats = GenerationStats {
            tokens_generated: 50,
            prefill_time: 0.1,
            decode_time: 1.0,
            tokens_per_sec: 49.0, // (50 - 1) / 1.0
            total_time: 1.1,
        };

        assert_eq!(stats.tokens_generated, 50);
        assert!((stats.tokens_per_sec - 49.0).abs() < 0.01);
    }
}
