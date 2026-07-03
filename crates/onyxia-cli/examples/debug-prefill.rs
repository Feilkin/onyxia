//! Debug probe: per-position prefill argmax on the wgpu backend vs the
//! reference interpreter, for a real chat-templated prompt.
//!
//! `gpu != ref` at some position ⇒ kernel bug at whatever op differs;
//! `gpu == ref` but generation looks wrong ⇒ suspect lowering semantics
//! (both backends share it — compare against onnxruntime instead).
//!
//! ```sh
//! cargo run --release -p onyxia-cli --example debug-prefill
//! ```

use anyhow::{Context, Result, bail};
use onyxia_backend_wgpu::{GpuContext, GpuTensor, WgpuBackend};
use onyxia_cli::tokenizer::{ChatMessage, Tokenizer};
use onyxia_ir::interp::Tensor as IrTensor;
use onyxia_ir::{Backend, Bindings, DataType, Module, Session};
use std::path::PathBuf;

fn main() -> Result<()> {
    let model_path = PathBuf::from("models/gemma-3-270m-it-ONNX/onnx/model.onnx");
    let tokenizer_dir = PathBuf::from("models/gemma-3-270m-it-ONNX/");
    let prompt = "Why is the sky blue?";

    let mut tokenizer = Tokenizer::from_file(tokenizer_dir.join("tokenizer.json"))?;
    tokenizer = tokenizer.with_chat_template_file(tokenizer_dir.join("chat_template.jinja"))?;
    let templated = tokenizer.apply_chat_template(
        &[ChatMessage {
            role: "user".to_string(),
            content: prompt.to_string(),
        }],
        true,
    )?;
    let ids = tokenizer.encode(&templated, false)?;
    let seq = ids.len();
    eprintln!("prompt ids ({seq}): {ids:?}");

    eprintln!("lowering …");
    let graph = onyxia_onnx::load_and_parse_model(&model_path)?;
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())?;
    let host_inputs = build_prefill_inputs(&module, &ids)?;

    eprintln!("wgpu backend: full prefill …");
    let gpu_logits = {
        let ctx = GpuContext::new_blocking()?;
        let backend = WgpuBackend::new(ctx);
        let mut session = backend.prepare(module.clone())?;
        pollster::block_on(async {
            let dev: Vec<(&str, GpuTensor)> = host_inputs
                .iter()
                .map(|(n, t)| Ok((n.as_str(), session.upload(t)?)))
                .collect::<onyxia_ir::Result<_>>()?;
            let outs = session.run(&dev).await?;
            let (_, logits) = outs
                .iter()
                .find(|(n, _)| n == "logits")
                .ok_or_else(|| onyxia_ir::Error::Runtime("no logits".into()))?;
            session.download(logits).await
        })?
    };

    eprintln!("reference interpreter: full prefill (slow) …");
    let input_refs: Vec<(&str, IrTensor)> = host_inputs
        .iter()
        .map(|(n, t)| (n.as_str(), t.clone()))
        .collect();
    let ref_outs = onyxia_backend_ref::run_once(module, &input_refs)?;
    let (_, ref_logits) = ref_outs
        .iter()
        .find(|(n, _)| n == "logits")
        .context("no logits in reference outputs")?;

    let gpu = per_position_argmax(&gpu_logits, seq)?;
    let rf = per_position_argmax(ref_logits, seq)?;
    println!("\npos | token(id)      | gpu      | ref      | verdict");
    let mut mismatches = 0;
    for i in 0..seq {
        let tok = tokenizer
            .decode(&[ids[i]], false)
            .unwrap_or_else(|_| "?".into());
        let ok = gpu[i] == rf[i];
        mismatches += usize::from(!ok);
        println!(
            "{i:3} | {tok:>10.10}({:>6}) | {:>8} | {:>8} | {}",
            ids[i],
            gpu[i],
            rf[i],
            if ok { "ok" } else { "MISMATCH" }
        );
    }
    if mismatches > 0 {
        bail!("{mismatches} position(s) disagree between GPU and reference");
    }
    Ok(())
}

fn argmax(v: &[f32]) -> usize {
    v.iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| i)
        .unwrap()
}

fn per_position_argmax(logits: &IrTensor, seq: usize) -> Result<Vec<usize>> {
    let all = logits.to_f32()?;
    let vocab = *logits.shape().last().context("scalar logits")?;
    Ok((0..seq)
        .map(|i| argmax(&all[i * vocab..(i + 1) * vocab]))
        .collect())
}

/// Host-side prefill inputs (past=0, dense mask).
fn build_prefill_inputs(module: &Module, ids: &[i64]) -> Result<Vec<(String, IrTensor)>> {
    let seq = ids.len();
    let mut bindings = Bindings::new();
    for (name, value) in [
        ("batch_size", 1u64),
        ("sequence_length", seq as u64),
        ("past_sequence_length", 0),
        ("total_sequence_length", seq as u64),
    ] {
        if let Some(sym) = module.symbols.get(name) {
            bindings.bind(sym, value)?;
        }
    }
    module
        .inputs
        .iter()
        .map(|(name, id)| {
            let ty = &module.value(*id).ty;
            let dims = ty.shape.eval(&bindings)?;
            let numel: usize = dims.iter().product();
            let tensor = match (name.as_str(), ty.dtype) {
                ("input_ids", DataType::I64) => IrTensor::from_i64(ids, &dims)?,
                ("position_ids", DataType::I64) => {
                    IrTensor::from_i64(&(0..numel as i64).collect::<Vec<_>>(), &dims)?
                }
                (_, DataType::I64) => IrTensor::from_i64(&vec![1; numel], &dims)?,
                (_, DataType::F32) => IrTensor::from_f32(&vec![0.0; numel], &dims)?,
                (n, dt) => bail!("unexpected input '{n}' dtype {dt}"),
            };
            Ok((name.clone(), tensor))
        })
        .collect()
}
