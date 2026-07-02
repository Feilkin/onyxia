//! C4 model parity gate (`doc/ir-implementation-plan.md`).
//!
//! Runs the same greedy generation twice — once through the old pipeline
//! (onyxia-compiler + onyxia-runtime `LlmSession`) and once through the new
//! one (onyxia-lower + onyxia-backend-wgpu `Session`) — and checks:
//!
//! 1. token-identical output for `--tokens` (default 64) tokens, and
//! 2. new-pipeline decode tokens/sec ≥ 90% of the old pipeline's.
//!
//! Both sides round-trip the KV cache through host memory so the comparison
//! is apples-to-apples (device-resident KV is milestone D). `--device-kv`
//! keeps the new side's KV on-device as a preview of D — informative, not
//! part of the gate.
//!
//! ```sh
//! cargo run --release -p onyxia-cli --example parity-gate -- \
//!     [models/gemma-3-270m-it-ONNX/onnx/model.onnx] [--tokens 64] [--device-kv]
//! ```

use anyhow::{Context, Result, bail};
use onyxia_backend_wgpu::{GpuContext, GpuTensor, WgpuBackend, WgpuSession};
use onyxia_cli::llm::{LlmConfig, LlmSession};
use onyxia_cli::sampling::{SamplingConfig, sample};
use onyxia_cli::tokenizer::{ChatMessage, Tokenizer};
use onyxia_ir::interp::Tensor as IrTensor;
use onyxia_ir::{Backend, Bindings, DataType, Module, Session};
use rand::SeedableRng;
use rand::rngs::StdRng;
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::time::Instant;

const MAX_SEQ_LEN: usize = 1024;

struct RunResult {
    tokens: Vec<u32>,
    prepare_time: f64,
    prefill_time: f64,
    decode_time: f64,
}

impl RunResult {
    fn tokens_per_sec(&self) -> f64 {
        (self.tokens.len().saturating_sub(1)) as f64 / self.decode_time
    }
}

fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let flag = |name: &str| args.iter().any(|a| a == name);
    let opt = |name: &str| {
        args.iter()
            .position(|a| a == name)
            .and_then(|i| args.get(i + 1).cloned())
    };

    let model_path = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-3-270m-it-ONNX/onnx/model.onnx"));
    let tokenizer_dir = opt("--tokenizer")
        .map(PathBuf::from)
        .unwrap_or_else(|| PathBuf::from("models/gemma-3-270m-it-ONNX/"));
    let prompt = opt("--prompt").unwrap_or_else(|| "Why is the sky blue?".to_string());
    let n_tokens: usize = opt("--tokens").map(|s| s.parse()).transpose()?.unwrap_or(64);
    let warmup: usize = opt("--warmup").map(|s| s.parse()).transpose()?.unwrap_or(8);
    let device_kv = flag("--device-kv");

    // Tokenize once; both pipelines see the same ids.
    let tokenizer_file = tokenizer_dir.join("tokenizer.json");
    let mut tokenizer = Tokenizer::from_file(&tokenizer_file)
        .with_context(|| format!("loading tokenizer from {}", tokenizer_file.display()))?;
    let chat_template = tokenizer_dir.join("chat_template.jinja");
    let prompt_for_model = if chat_template.exists() {
        tokenizer = tokenizer.with_chat_template_file(&chat_template)?;
        tokenizer.apply_chat_template(
            &[ChatMessage {
                role: "user".to_string(),
                content: prompt.clone(),
            }],
            true,
        )?
    } else {
        prompt.clone()
    };
    let prompt_ids = tokenizer.encode(&prompt_for_model, false)?;
    println!(
        "prompt: {prompt:?} ({} tokens), generating {n_tokens} tokens greedily\n",
        prompt_ids.len()
    );

    // Run sequentially and drop each pipeline before the next so both fit in
    // VRAM (each holds its own copy of the weights).
    println!("=== old pipeline (onyxia-compiler + onyxia-runtime) ===");
    let old = run_old(&model_path, &prompt_ids, n_tokens, warmup)?;
    report("old", &old, &tokenizer);

    println!(
        "\n=== new pipeline (onyxia-lower + onyxia-backend-wgpu{}) ===",
        if device_kv { ", device-resident KV" } else { "" }
    );
    let new = run_new(&model_path, &prompt_ids, n_tokens, warmup, device_kv)?;
    report("new", &new, &tokenizer);

    // Verdict.
    println!("\n=== C4 gate ===");
    let divergence = old
        .tokens
        .iter()
        .zip(&new.tokens)
        .position(|(a, b)| a != b);
    let identical = divergence.is_none() && old.tokens.len() == new.tokens.len();
    match divergence {
        None => println!(
            "tokens: identical for {} tokens {}",
            old.tokens.len(),
            if identical { "✓" } else { "(length mismatch!)" }
        ),
        Some(i) => println!(
            "tokens: FIRST DIVERGENCE at step {i}: old={} new={}",
            old.tokens[i], new.tokens[i]
        ),
    }
    let ratio = new.tokens_per_sec() / old.tokens_per_sec();
    println!(
        "decode: old {:.2} tok/s, new {:.2} tok/s (ratio {ratio:.2})",
        old.tokens_per_sec(),
        new.tokens_per_sec()
    );
    let token_gate = identical && old.tokens.len() >= 64;
    let perf_gate = ratio >= 0.9;
    println!(
        "GATE token-identical (≥64): {}",
        if token_gate { "PASS" } else { "FAIL" }
    );
    println!(
        "GATE tokens/sec within 10%: {}",
        if perf_gate { "PASS" } else { "FAIL" }
    );
    if !(token_gate && perf_gate) {
        std::process::exit(1);
    }
    Ok(())
}

fn report(label: &str, r: &RunResult, tokenizer: &Tokenizer) {
    let ids: Vec<i64> = r.tokens.iter().map(|&t| t as i64).collect();
    let text = tokenizer.decode(&ids, false).unwrap_or_default();
    println!(
        "{label}: prepare {:.2}s, prefill {:.3}s, decode {:.3}s → {:.2} tok/s",
        r.prepare_time,
        r.prefill_time,
        r.decode_time,
        r.tokens_per_sec()
    );
    println!("{label} text: {text:?}");
}

// ---------------------------------------------------------------- old side

fn run_old(model: &Path, prompt_ids: &[i64], n: usize, warmup: usize) -> Result<RunResult> {
    let started = Instant::now();
    let graph = onyxia_onnx::load_and_parse_model(model)?;
    let runtime = pollster::block_on(onyxia_runtime::Runtime::new())?;
    let registry = onyxia_operators::core_operator_registry();
    let mut pipeline = onyxia_compiler::CompilerPipeline::new();
    let compiled = pollster::block_on(pipeline.compile(&graph, &registry, runtime.gpu()))?;
    let executor = runtime.load_model(compiled)?;
    let mut session = LlmSession::new(
        executor,
        &LlmConfig {
            max_seq_len: MAX_SEQ_LEN,
            num_layers: 0, // unused by KV discovery
        },
    );
    let prepare_time = started.elapsed().as_secs_f64();

    if warmup > 0 {
        greedy_old(&mut session, prompt_ids, warmup.min(n))?;
        session.reset_full();
    }
    let (tokens, prefill_time, decode_time) = greedy_old(&mut session, prompt_ids, n)?;
    Ok(RunResult {
        tokens,
        prepare_time,
        prefill_time,
        decode_time,
    })
}

fn greedy_old(session: &mut LlmSession, prompt_ids: &[i64], n: usize) -> Result<(Vec<u32>, f64, f64)> {
    let (cfg, mut rng) = greedy_sampler();
    let t0 = Instant::now();
    let logits = session.prefill(prompt_ids)?;
    let prefill = t0.elapsed().as_secs_f64();
    let mut tokens = vec![sample(&logits, &cfg, &mut rng)];
    let t1 = Instant::now();
    for _ in 1..n {
        let logits = session.decode(*tokens.last().unwrap() as i64)?;
        tokens.push(sample(&logits, &cfg, &mut rng));
    }
    Ok((tokens, prefill, t1.elapsed().as_secs_f64()))
}

/// Greedy = argmax; both pipelines share this so tie-breaking is identical.
fn greedy_sampler() -> (SamplingConfig, StdRng) {
    (
        SamplingConfig {
            temperature: 0.0,
            top_k: 0,
            top_p: 0.0,
            seed: Some(0),
        },
        StdRng::seed_from_u64(0),
    )
}

// ---------------------------------------------------------------- new side

fn run_new(
    model: &Path,
    prompt_ids: &[i64],
    n: usize,
    warmup: usize,
    device_kv: bool,
) -> Result<RunResult> {
    let started = Instant::now();
    let graph = onyxia_onnx::load_and_parse_model(model)?;
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())?;
    let ctx = GpuContext::new_blocking()?;
    let backend = WgpuBackend::new(ctx);
    let session = backend.prepare(module.clone())?;
    let mut llm = NewLlm::new(session, module, device_kv);
    let prepare_time = started.elapsed().as_secs_f64();

    pollster::block_on(async {
        if warmup > 0 {
            greedy_new(&mut llm, prompt_ids, warmup.min(n)).await?;
            llm.reset();
        }
        let (tokens, prefill_time, decode_time) = greedy_new(&mut llm, prompt_ids, n).await?;
        Ok(RunResult {
            tokens,
            prepare_time,
            prefill_time,
            decode_time,
        })
    })
}

async fn greedy_new(llm: &mut NewLlm, prompt_ids: &[i64], n: usize) -> Result<(Vec<u32>, f64, f64)> {
    let (cfg, mut rng) = greedy_sampler();
    let positions: Vec<i64> = (0..prompt_ids.len() as i64).collect();
    let t0 = Instant::now();
    let logits = llm.step(prompt_ids, &positions).await?;
    let prefill = t0.elapsed().as_secs_f64();
    let mut tokens = vec![sample(&logits, &cfg, &mut rng)];
    let t1 = Instant::now();
    for _ in 1..n {
        let token = *tokens.last().unwrap() as i64;
        let logits = llm.step(&[token], &[llm.past_len as i64]).await?;
        tokens.push(sample(&logits, &cfg, &mut rng));
    }
    Ok((tokens, prefill, t1.elapsed().as_secs_f64()))
}

/// Minimal LLM loop over the new `Session` API, mirroring the old
/// `LlmSession`: `present.*` outputs are stored and fed back as
/// `past_key_values.*` inputs (through host memory by default, staying
/// on-device with `--device-kv`).
struct NewLlm {
    session: WgpuSession,
    module: Module,
    /// `(present output name, past input name)` pairs.
    kv_pairs: Vec<(String, String)>,
    kv_host: HashMap<String, IrTensor>,
    kv_dev: HashMap<String, GpuTensor>,
    device_kv: bool,
    past_len: usize,
}

impl NewLlm {
    fn new(session: WgpuSession, module: Module, device_kv: bool) -> Self {
        let input_names: std::collections::HashSet<&str> =
            module.inputs.iter().map(|(n, _)| n.as_str()).collect();
        let kv_pairs = module
            .outputs
            .iter()
            .filter_map(|(out, _)| {
                let rest = out.strip_prefix("present.")?;
                let past = format!("past_key_values.{rest}");
                input_names
                    .contains(past.as_str())
                    .then(|| (out.clone(), past))
            })
            .collect();
        Self {
            session,
            module,
            kv_pairs,
            kv_host: HashMap::new(),
            kv_dev: HashMap::new(),
            device_kv,
            past_len: 0,
        }
    }

    fn reset(&mut self) {
        self.kv_host.clear();
        self.kv_dev.clear();
        self.past_len = 0;
    }

    /// One forward pass over `ids` (prefill when `ids.len() > 1`, decode when
    /// 1). Returns the logits of the last position.
    async fn step(&mut self, ids: &[i64], positions: &[i64]) -> Result<Vec<f32>> {
        let seq = ids.len();
        let bindings = self.bindings_for(seq)?;

        let mut inputs: Vec<(String, GpuTensor)> = Vec::new();
        for (name, id) in self.module.inputs.clone() {
            let ty = self.module.value(id).ty.clone();
            let tensor = if name == "input_ids" {
                self.session.upload(&IrTensor::from_i64(ids, &[1, seq])?)?
            } else if name == "position_ids" {
                self.session
                    .upload(&IrTensor::from_i64(positions, &[1, seq])?)?
            } else if name.starts_with("past_key_values.") {
                if let Some(t) = self.kv_dev.get(&name) {
                    t.clone()
                } else if let Some(host) = self.kv_host.get(&name) {
                    self.session.upload(host)?
                } else {
                    // First step: empty cache in the declared shape (past=0).
                    let dims = ty.shape.eval(&bindings)?;
                    let numel: usize = dims.iter().product();
                    match ty.dtype {
                        DataType::F32 => self
                            .session
                            .upload(&IrTensor::from_f32(&vec![0.0; numel], &dims)?)?,
                        dt => bail!("KV input '{name}' has unsupported dtype {dt}"),
                    }
                }
            } else {
                // Masks and friends: all-ones in the declared shape.
                let dims = ty.shape.eval(&bindings)?;
                let numel: usize = dims.iter().product();
                match ty.dtype {
                    DataType::I64 => self
                        .session
                        .upload(&IrTensor::from_i64(&vec![1; numel], &dims)?)?,
                    DataType::F32 => self
                        .session
                        .upload(&IrTensor::from_f32(&vec![1.0; numel], &dims)?)?,
                    dt => bail!("input '{name}' has unsupported dtype {dt}"),
                }
            };
            inputs.push((name, tensor));
        }

        let refs: Vec<(&str, GpuTensor)> = inputs
            .iter()
            .map(|(n, t)| (n.as_str(), t.clone()))
            .collect();
        let outputs = self.session.run(&refs).await?;

        for (present, past) in self.kv_pairs.clone() {
            let (_, t) = outputs
                .iter()
                .find(|(n, _)| *n == present)
                .with_context(|| format!("output '{present}' missing"))?;
            if self.device_kv {
                self.kv_dev.insert(past, t.clone());
            } else {
                self.kv_host.insert(past, self.session.download(t).await?);
            }
        }
        self.past_len += seq;

        let (_, logits) = outputs
            .iter()
            .find(|(n, _)| n == "logits")
            .context("output 'logits' missing")?;
        let logits = self.session.download(logits).await?;
        let vocab = *logits.shape().last().context("scalar logits")?;
        let all = logits.to_f32()?;
        Ok(all[(seq - 1) * vocab..seq * vocab].to_vec())
    }

    fn bindings_for(&self, seq: usize) -> Result<Bindings> {
        let mut bindings = Bindings::new();
        for (name, value) in [
            ("batch_size", 1u64),
            ("sequence_length", seq as u64),
            ("past_sequence_length", self.past_len as u64),
            ("total_sequence_length", (self.past_len + seq) as u64),
        ] {
            if let Some(sym) = self.module.symbols.get(name) {
                bindings.bind(sym, value)?;
            }
        }
        Ok(bindings)
    }
}
