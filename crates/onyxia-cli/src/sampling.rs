//! Token sampling strategies for LLM generation.
//!
//! Implements various methods for selecting the next token from model logits:
//! - Greedy (argmax) - deterministic selection of highest probability token
//! - Top-K - sample from top K tokens by probability
//! - Top-P (nucleus) - sample from smallest set of tokens with cumulative probability >= p
//!
//! All sampling methods support temperature scaling for controlling randomness.

use rand::prelude::*;

/// Configuration for sampling strategies.
#[derive(Debug, Clone)]
pub struct SamplingConfig {
    /// Temperature scaling (0.0 = greedy, 1.0 = no scaling, >1.0 = more random).
    pub temperature: f32,
    /// Top-K sampling: keep only top K tokens (0 = disabled).
    pub top_k: usize,
    /// Top-P (nucleus) sampling: keep tokens with cumulative probability >= p (0.0 = disabled, 1.0 = all tokens).
    pub top_p: f32,
    /// Random seed for reproducible sampling (None = use system entropy).
    pub seed: Option<u64>,
}

impl Default for SamplingConfig {
    fn default() -> Self {
        Self {
            temperature: 1.0,
            top_k: 0,   // disabled
            top_p: 0.0, // disabled
            seed: None, // non-deterministic
        }
    }
}

/// Sample the next token from logits according to the configuration.
///
/// # Arguments
/// * `logits` - Raw logits from the model \[vocab_size\]
/// * `config` - Sampling configuration
/// * `rng` - Random number generator (for reproducibility)
///
/// # Returns
/// Token ID (index into logits)
///
/// # Panics
/// Panics if logits is empty or contains all NaN/Inf values
pub fn sample(logits: &[f32], config: &SamplingConfig, rng: &mut StdRng) -> u32 {
    assert!(!logits.is_empty(), "Cannot sample from empty logits");

    // Check for invalid logits
    let valid_count = logits.iter().filter(|x| x.is_finite()).count();
    if valid_count == 0 {
        panic!("All logits are NaN or Inf, cannot sample");
    }

    // If temperature is 0.0, use greedy sampling
    if config.temperature <= 0.0 || config.temperature < 1e-6 {
        return sample_greedy(logits);
    }

    // Apply temperature scaling
    let mut scaled_logits: Vec<f32> = logits.iter().map(|&x| x / config.temperature).collect();

    // Apply top-k or top-p filtering, then sample
    if config.top_k > 0 {
        sample_top_k(&mut scaled_logits, config.top_k, rng)
    } else if config.top_p > 0.0 && config.top_p < 1.0 {
        sample_top_p(&mut scaled_logits, config.top_p, rng)
    } else {
        // No filtering, sample from full distribution
        sample_from_distribution(&scaled_logits, rng)
    }
}

/// Greedy sampling: select the token with the highest logit value (argmax).
///
/// # Arguments
/// * `logits` - Raw logits from the model \\[vocab_size\\]
///
/// # Returns
/// Token ID with highest probability
///
/// # Panics
/// Panics if logits is empty
pub fn sample_greedy(logits: &[f32]) -> u32 {
    assert!(!logits.is_empty(), "Cannot sample_greedy from empty logits");

    logits
        .iter()
        .enumerate()
        .max_by(|(_, a), (_, b)| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
        .map(|(idx, _)| idx as u32)
        .expect("max_by should never return None for non-empty iterator")
}

/// Top-K sampling: sample from the top K tokens by probability.
///
/// # Arguments
/// * `logits` - Temperature-scaled logits \\[vocab_size\\] (will be modified in-place)
/// * `k` - Number of top tokens to keep
/// * `rng` - Random number generator
///
/// # Returns
/// Sampled token ID
pub fn sample_top_k(logits: &mut [f32], k: usize, rng: &mut StdRng) -> u32 {
    if k >= logits.len() {
        return sample_from_distribution(logits, rng);
    }

    // Find indices of top-k elements using partial sort
    let mut indices: Vec<usize> = (0..logits.len()).collect();
    indices.select_nth_unstable_by(k, |&a, &b| {
        logits[b]
            .partial_cmp(&logits[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });

    // Keep only top-k, zero out the rest
    let top_k_indices = &indices[..k];
    let mut filtered_logits = vec![f32::NEG_INFINITY; logits.len()];
    for &idx in top_k_indices {
        filtered_logits[idx] = logits[idx];
    }

    sample_from_distribution(&filtered_logits, rng)
}

/// Top-P (nucleus) sampling: sample from the smallest set of tokens with cumulative probability >= p.
///
/// # Arguments
/// * `logits` - Temperature-scaled logits \\[vocab_size\\] (will be modified in-place)
/// * `p` - Cumulative probability threshold (e.g., 0.9)
/// * `rng` - Random number generator
///
/// # Returns
/// Sampled token ID
pub fn sample_top_p(logits: &mut [f32], p: f32, rng: &mut StdRng) -> u32 {
    // Sort indices by descending logit value
    let mut indexed: Vec<(usize, f32)> = logits.iter().copied().enumerate().collect();
    indexed.sort_by(|(_, a), (_, b)| b.partial_cmp(a).unwrap_or(std::cmp::Ordering::Equal));

    // Compute softmax and find cutoff index where cumulative probability >= p
    let max_logit = indexed[0].1;
    let mut sum_exp = 0.0f32;
    let mut exp_vals: Vec<f32> = indexed
        .iter()
        .map(|(_, logit)| {
            let exp_val = (logit - max_logit).exp();
            sum_exp += exp_val;
            exp_val
        })
        .collect();

    // Normalize to probabilities
    for exp_val in &mut exp_vals {
        *exp_val /= sum_exp;
    }

    // Find cutoff where cumulative probability >= p
    let mut cumsum = 0.0f32;
    let mut cutoff = indexed.len();
    for (i, &prob) in exp_vals.iter().enumerate() {
        cumsum += prob;
        if cumsum >= p {
            cutoff = i + 1;
            break;
        }
    }

    // Keep only tokens up to cutoff
    let mut filtered_logits = vec![f32::NEG_INFINITY; logits.len()];
    for (idx, logit) in indexed.iter().take(cutoff) {
        filtered_logits[*idx] = *logit;
    }

    sample_from_distribution(&filtered_logits, rng)
}

/// Sample from a probability distribution defined by logits.
///
/// Applies softmax to logits and samples according to the resulting probabilities.
///
/// # Arguments
/// * `logits` - Temperature-scaled logits \[vocab_size\]
/// * `rng` - Random number generator
///
/// # Returns
/// Sampled token ID
fn sample_from_distribution(logits: &[f32], rng: &mut StdRng) -> u32 {
    // Compute softmax with numerical stability (subtract max)
    let max_logit = logits.iter().copied().fold(f32::NEG_INFINITY, f32::max);

    let mut sum_exp = 0.0f32;
    let exp_vals: Vec<f32> = logits
        .iter()
        .map(|&logit| {
            if logit.is_finite() {
                let exp_val = (logit - max_logit).exp();
                sum_exp += exp_val;
                exp_val
            } else {
                0.0
            }
        })
        .collect();

    // Sample from categorical distribution
    let rand_u32 = rng.next_u32();
    let rand_val = (rand_u32 as f32 / u32::MAX as f32) * sum_exp;
    let mut cumsum = 0.0f32;
    for (idx, &exp_val) in exp_vals.iter().enumerate() {
        cumsum += exp_val;
        if cumsum >= rand_val {
            return idx as u32;
        }
    }

    // Fallback (should never reach here)
    (exp_vals.len() - 1) as u32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_sample_greedy() {
        let logits = vec![0.1, 0.5, 0.8, 0.3];
        assert_eq!(sample_greedy(&logits), 2);

        let logits = vec![-1.0, -0.1, -0.5];
        assert_eq!(sample_greedy(&logits), 1);
    }

    #[test]
    fn test_sample_greedy_with_negatives() {
        let logits = vec![-10.0, -5.0, -1.0, -20.0];
        assert_eq!(sample_greedy(&logits), 2);
    }

    #[test]
    fn test_sample_temperature() {
        let logits = vec![1.0, 2.0, 3.0];
        let mut rng = StdRng::seed_from_u64(42);

        // Temperature = 0 should be greedy
        let config = SamplingConfig {
            temperature: 0.0,
            ..Default::default()
        };
        assert_eq!(sample(&logits, &config, &mut rng), 2);

        // Higher temperature should sample from distribution
        let config = SamplingConfig {
            temperature: 2.0,
            ..Default::default()
        };
        let token = sample(&logits, &config, &mut rng);
        assert!(token <= 2);
    }

    #[test]
    fn test_sample_top_k() {
        let mut logits = vec![0.1, 0.5, 0.8, 0.3, 0.2];
        let mut rng = StdRng::seed_from_u64(42);

        // Top-2 should only sample from indices 1 and 2 (logits 0.5 and 0.8)
        let token = sample_top_k(&mut logits, 2, &mut rng);
        assert!(token == 1 || token == 2);
    }

    #[test]
    fn test_sample_top_p() {
        let mut logits = vec![1.0, 0.5, 0.1, 0.05];
        let mut rng = StdRng::seed_from_u64(42);

        // With small p, should only sample from top tokens
        let token = sample_top_p(&mut logits, 0.5, &mut rng);
        assert!(token <= 1); // Should be 0 or 1
    }

    #[test]
    fn test_sample_deterministic_seed() {
        let logits = vec![1.0, 2.0, 1.5, 0.5];

        let config = SamplingConfig {
            temperature: 1.0,
            seed: Some(12345),
            ..Default::default()
        };

        let mut rng1 = StdRng::seed_from_u64(12345);
        let mut rng2 = StdRng::seed_from_u64(12345);

        let token1 = sample(&logits, &config, &mut rng1);
        let token2 = sample(&logits, &config, &mut rng2);

        assert_eq!(token1, token2, "Same seed should produce same token");
    }

    #[test]
    fn test_sample_from_distribution() {
        let logits = vec![0.0, 1.0, 2.0];
        let mut rng = StdRng::seed_from_u64(999);

        // Run multiple samples to ensure we get valid tokens
        for _ in 0..100 {
            let token = sample_from_distribution(&logits, &mut rng);
            assert!(token <= 2);
        }
    }
}
