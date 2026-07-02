//! Whole-model gate for the CubeCL backend: run one prefill forward pass of
//! a real model on BOTH backends (wgpu and cubecl) and compare logits +
//! timings. The cubecl backend executes the model purely as primitives —
//! every composite (GQA, RMS-norm, Softmax, Rotary, Gelu) inlines through
//! its decomposition.
//!
//! ```sh
//! cargo run --release -p onyxia-backend-cubecl --example forward-check -- \
//!     models/gemma-3-270m-it-ONNX/onnx/model.onnx [--seq 15]
//! ```

use onyxia_ir::interp::Tensor;
use onyxia_ir::{Backend, Bindings, DataType, Module, Session};
use std::time::Instant;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .filter(|a| !a.starts_with("--"))
        .cloned()
        .unwrap_or_else(|| "models/gemma-3-270m-it-ONNX/onnx/model.onnx".to_string());
    let seq: usize = args
        .iter()
        .position(|a| a == "--seq")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(15);

    eprintln!("parsing + lowering {path} …");
    let graph = onyxia_onnx::load_and_parse_model(&path)?;
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())?;
    let inputs = build_inputs(&module, seq)?;

    // ── wgpu backend (fused composites + generated WGSL) ────────────────
    let wgpu_logits = {
        eprintln!("\n=== wgpu backend ===");
        let started = Instant::now();
        let ctx = onyxia_backend_wgpu::GpuContext::new_blocking()?;
        let backend = onyxia_backend_wgpu::WgpuBackend::new(ctx);
        let mut session = backend.prepare(module.clone())?;
        eprintln!("prepared in {:.2?}", started.elapsed());
        run_prefill(&mut session, &inputs, seq)?
    };

    // ── cubecl backend (primitives only) ─────────────────────────────────
    let cube_logits = {
        eprintln!("\n=== cubecl backend (primitives only) ===");
        let started = Instant::now();
        let backend = onyxia_backend_cubecl::CubeclBackend::new();
        let mut session = backend.prepare(module.clone())?;
        eprintln!("prepared in {:.2?}", started.elapsed());
        run_prefill(&mut session, &inputs, seq)?
    };

    // ── compare ───────────────────────────────────────────────────────────
    let mut max_abs = 0f32;
    for (a, b) in wgpu_logits.iter().zip(&cube_logits) {
        max_abs = max_abs.max((a - b).abs());
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    println!("\nmax |Δlogit| (last position, wgpu vs cubecl): {max_abs}");
    println!(
        "argmax: wgpu={} cubecl={} {}",
        argmax(&wgpu_logits),
        argmax(&cube_logits),
        if argmax(&wgpu_logits) == argmax(&cube_logits) {
            "✓ MATCH"
        } else {
            "✗ MISMATCH"
        }
    );
    Ok(())
}

/// Cold + warm prefill; returns last-position logits from the warm pass.
fn run_prefill<S: Session>(
    session: &mut S,
    inputs: &[(String, Tensor)],
    seq: usize,
) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    pollster::block_on(async {
        let dev: Vec<(&str, S::Tensor)> = inputs
            .iter()
            .map(|(n, t)| Ok((n.as_str(), session.upload(t)?)))
            .collect::<onyxia_ir::Result<_>>()?;
        let cold = Instant::now();
        let outs = session.run(&dev).await?;
        let logits = find(&outs, "logits")?;
        let host = session.download(logits).await?;
        eprintln!("prefill (cold, incl. JIT) {:.2?}", cold.elapsed());
        let warm = Instant::now();
        let outs = session.run(&dev).await?;
        let logits = find(&outs, "logits")?;
        let host2 = session.download(logits).await?;
        eprintln!("prefill (warm)            {:.2?}", warm.elapsed());
        let _ = host;
        last_position(&host2, seq)
    })
}

fn find<'o, T>(
    outs: &'o [(String, T)],
    name: &str,
) -> Result<&'o T, Box<dyn std::error::Error>> {
    outs.iter()
        .find(|(n, _)| n == name)
        .map(|(_, t)| t)
        .ok_or_else(|| format!("no '{name}' output").into())
}

/// Logits for the last sequence position: `[1, S, V] → [V]`.
fn last_position(logits: &Tensor, seq: usize) -> Result<Vec<f32>, Box<dyn std::error::Error>> {
    let all = logits.to_f32()?;
    let vocab = logits.shape()[logits.shape().len() - 1];
    Ok(all[(seq - 1) * vocab..seq * vocab].to_vec())
}

/// Prefill inputs bound with S=seq, past=0 (BOS + fixed ids; dense mask).
fn build_inputs(module: &Module, seq: usize) -> Result<Vec<(String, Tensor)>, onyxia_ir::Error> {
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
                ("input_ids", DataType::I64) => {
                    let ids: Vec<i64> = (0..numel)
                        .map(|i| if i == 0 { 2 } else { 651 + i as i64 })
                        .collect();
                    Tensor::from_i64(&ids, &dims)?
                }
                ("position_ids", DataType::I64) => {
                    Tensor::from_i64(&(0..numel as i64).collect::<Vec<_>>(), &dims)?
                }
                (_, DataType::I64) => Tensor::from_i64(&vec![1; numel], &dims)?,
                (_, DataType::F32) => Tensor::from_f32(&vec![0.0; numel], &dims)?,
                (n, dt) => {
                    return Err(onyxia_ir::Error::Unsupported(format!(
                        "input '{n}' dtype {dt} in forward-check"
                    )));
                }
            };
            Ok((name.clone(), tensor))
        })
        .collect()
}
