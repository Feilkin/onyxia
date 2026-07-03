//! LLM-specific session management with KV cache support.
//!
//! Wraps a wgpu-backend [`Session`] (onyxia-lower + onyxia-backend-wgpu) and
//! manages the KV cache **on-device**: `present.*` output handles are stored
//! and fed back as `past_key_values.*` inputs on the next call — no host
//! round-trip. Only the logits are downloaded.
//!
//! All methods are async so the same code runs on native (driven by a
//! blocking executor on a worker thread) and on the web (spawned on the
//! browser thread, yielding at every await — WebGPU readback cannot block).

use anyhow::{Context, Result, bail};
use onyxia_backend_wgpu::{GpuTensor, WgpuBackend, WgpuSession};
use onyxia_ir::interp::Tensor;
use onyxia_ir::{Backend, Bindings, DataType, Module, Session, SymbolTable, SymbolicShape};
use std::collections::HashMap;

/// One module input: everything needed to construct a bound tensor for it.
struct InputSpec {
    name: String,
    dtype: DataType,
    shape: SymbolicShape,
}

/// High-level LLM inference session with device-resident KV cache.
pub struct LlmSession {
    session: WgpuSession,
    inputs: Vec<InputSpec>,
    symbols: SymbolTable,
    /// KV cache tensor name pairs: (present_output_name, past_input_name).
    kv_pairs: Vec<(String, String)>,
    /// Device-resident KV handles, keyed by past_key_values.* name.
    kv_cache: HashMap<String, GpuTensor>,
    /// Current past sequence length (grows each call).
    past_seq_len: usize,
    /// Hard bound on total sequence length (clean error, not a GPU fault).
    max_seq_len: usize,
}

impl LlmSession {
    /// Prepare `module` on `backend` and set up KV-cache plumbing.
    ///
    /// Input specs and KV pairs are extracted before `prepare` consumes the
    /// module, so the (GiB-sized) constant pool is never cloned — that
    /// matters on wasm, where address space tops out at 4 GB.
    pub fn new(backend: &WgpuBackend, module: Module, max_seq_len: usize) -> Result<Self> {
        let inputs: Vec<InputSpec> = module
            .inputs
            .iter()
            .map(|(name, id)| {
                let ty = &module.value(*id).ty;
                InputSpec {
                    name: name.clone(),
                    dtype: ty.dtype,
                    shape: ty.shape.clone(),
                }
            })
            .collect();
        let kv_pairs = discover_kv_pairs(&module);
        let symbols = module.symbols.clone();
        let session = backend
            .prepare(module)
            .context("Failed to prepare model on the wgpu backend")?;
        Ok(Self {
            session,
            inputs,
            symbols,
            kv_pairs,
            kv_cache: HashMap::new(),
            past_seq_len: 0,
            max_seq_len,
        })
    }

    /// Bytes of GPU memory resident for this session: weights, the
    /// device-resident KV cache, and the backend's live buffer pool.
    /// Grows with context length as the KV cache does.
    pub fn vram_bytes(&self) -> u64 {
        self.session.resident_bytes()
    }

    /// Prefill: process the full prompt. Returns logits \[vocab_size\] for
    /// the last position.
    pub async fn prefill(&mut self, input_ids: &[i64]) -> Result<Vec<f32>> {
        if input_ids.is_empty() {
            bail!("Cannot prefill with empty input_ids");
        }
        if input_ids.len() > self.max_seq_len {
            bail!(
                "Input length {} exceeds max_seq_len {}",
                input_ids.len(),
                self.max_seq_len
            );
        }
        let positions: Vec<i64> = (0..input_ids.len() as i64).collect();
        self.step(input_ids, &positions).await
    }

    /// Decode: generate one next-token logit vector \[vocab_size\].
    pub async fn decode(&mut self, token_id: i64) -> Result<Vec<f32>> {
        if self.past_seq_len >= self.max_seq_len {
            bail!(
                "Sequence length {} would exceed max_seq_len {}",
                self.past_seq_len + 1,
                self.max_seq_len
            );
        }
        let position = self.past_seq_len as i64;
        self.step(&[token_id], &[position]).await
    }

    /// One forward pass over `ids`. Uploads bound inputs, feeds stored KV
    /// handles back, stores the new `present.*` handles, downloads logits.
    async fn step(&mut self, ids: &[i64], positions: &[i64]) -> Result<Vec<f32>> {
        let seq = ids.len();
        let bindings = self.bindings_for(seq)?;

        let mut inputs: Vec<(String, GpuTensor)> = Vec::with_capacity(self.inputs.len());
        for i in 0..self.inputs.len() {
            let (name, dtype) = (self.inputs[i].name.clone(), self.inputs[i].dtype);
            let tensor = if name == "input_ids" {
                self.session.upload(&Tensor::from_i64(ids, &[1, seq])?)?
            } else if name == "position_ids" {
                self.session
                    .upload(&Tensor::from_i64(positions, &[1, seq])?)?
            } else if name.starts_with("past_key_values.") {
                if let Some(t) = self.kv_cache.get(&name) {
                    t.clone()
                } else {
                    // First step: empty cache in the declared shape (past=0).
                    let dims = self.inputs[i].shape.eval(&bindings)?;
                    let numel: usize = dims.iter().product();
                    match dtype {
                        DataType::F32 => self
                            .session
                            .upload(&Tensor::from_f32(&vec![0.0; numel], &dims)?)?,
                        dt => bail!("KV input '{name}' has unsupported dtype {dt}"),
                    }
                }
            } else {
                // Masks and friends: all-ones in the declared shape.
                let dims = self.inputs[i].shape.eval(&bindings)?;
                let numel: usize = dims.iter().product();
                match dtype {
                    DataType::I64 => self
                        .session
                        .upload(&Tensor::from_i64(&vec![1; numel], &dims)?)?,
                    DataType::F32 => self
                        .session
                        .upload(&Tensor::from_f32(&vec![1.0; numel], &dims)?)?,
                    dt => bail!("input '{name}' has unsupported dtype {dt}"),
                }
            };
            inputs.push((name, tensor));
        }

        let refs: Vec<(&str, GpuTensor)> = inputs
            .iter()
            .map(|(n, t)| (n.as_str(), t.clone()))
            .collect();
        let outputs = self.session.run(&refs).await.with_context(|| {
            format!(
                "Model execution failed (seq={seq}, past={})",
                self.past_seq_len
            )
        })?;

        // Keep the new present.* handles on-device for the next call.
        for (present, past) in self.kv_pairs.clone() {
            let (_, t) = outputs
                .iter()
                .find(|(n, _)| *n == present)
                .with_context(|| format!("output '{present}' missing"))?;
            self.kv_cache.insert(past, t.clone());
        }
        self.past_seq_len += seq;

        // Download logits; return the last position ([1, S, V] → [V]).
        let (_, logits) = outputs
            .iter()
            .find(|(n, _)| n == "logits")
            .context("output 'logits' missing")?;
        let logits = self.session.download(logits).await?;
        let vocab = *logits.shape().last().context("scalar logits output")?;
        let all = logits.to_f32()?;
        Ok(all[(seq - 1) * vocab..seq * vocab].to_vec())
    }

    fn bindings_for(&self, seq: usize) -> Result<Bindings> {
        let mut bindings = Bindings::new();
        for (name, value) in [
            ("batch_size", 1u64),
            ("sequence_length", seq as u64),
            ("past_sequence_length", self.past_seq_len as u64),
            ("total_sequence_length", (self.past_seq_len + seq) as u64),
        ] {
            if let Some(sym) = self.symbols.get(name) {
                bindings.bind(sym, value)?;
            }
        }
        Ok(bindings)
    }

    /// Reset state and clear the KV cache for a completely fresh prefill.
    /// Required when re-prefilling a full conversation each turn.
    pub fn reset_full(&mut self) {
        self.past_seq_len = 0;
        self.kv_cache.clear();
    }
}

/// Discover KV cache pairs by name: `present.{N}.{key|value}` outputs paired
/// with `past_key_values.{N}.{key|value}` inputs.
fn discover_kv_pairs(module: &Module) -> Vec<(String, String)> {
    let input_names: std::collections::HashSet<&str> =
        module.inputs.iter().map(|(n, _)| n.as_str()).collect();
    module
        .outputs
        .iter()
        .filter_map(|(out, _)| {
            let rest = out.strip_prefix("present.")?;
            let past = format!("past_key_values.{rest}");
            input_names
                .contains(past.as_str())
                .then(|| (out.clone(), past))
        })
        .collect()
}
