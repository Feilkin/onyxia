//! Whole-model GPU gate: run one prefill forward pass of a real model on
//! the wgpu backend and (optionally) compare logits against the reference
//! backend.
//!
//! ```sh
//! cargo run --release -p onyxia-backend-wgpu --example forward-check -- \
//!     models/gemma-3-270m-it-ONNX/onnx/model.onnx [--seq 4] [--skip-ref]
//! ```

use onyxia_ir::interp::Tensor;
use onyxia_ir::{Backend, DataType, Module, Session};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().collect();
    let path = args
        .get(1)
        .ok_or("usage: forward-check <model.onnx> [--seq N] [--skip-ref]")?;
    let seq: usize = args
        .iter()
        .position(|a| a == "--seq")
        .and_then(|i| args.get(i + 1))
        .map(|s| s.parse())
        .transpose()?
        .unwrap_or(4);
    let skip_ref = args.iter().any(|a| a == "--skip-ref");

    eprintln!("parsing + lowering {path} …");
    let graph = onyxia_onnx::load_and_parse_model(path)?;
    let module = onyxia_lower::lower(graph, &onyxia_lower::standard_registry())?;

    let inputs = build_inputs(&module, seq)?;
    let input_refs: Vec<(&str, Tensor)> = inputs
        .iter()
        .map(|(n, t)| (n.as_str(), t.clone()))
        .collect();

    eprintln!("preparing wgpu session …");
    let started = std::time::Instant::now();
    let ctx = onyxia_backend_wgpu::GpuContext::new_blocking()?;
    let backend = onyxia_backend_wgpu::WgpuBackend::new(ctx);
    let mut session = backend.prepare(module.clone())?;
    eprintln!("prepared in {:.2?}", started.elapsed());

    eprintln!("running prefill (seq={seq}) on GPU …");
    let started = std::time::Instant::now();
    let logits_gpu = pollster::block_on(async {
        let dev: Vec<(&str, _)> = input_refs
            .iter()
            .map(|(n, t)| Ok((*n, session.upload(t)?)))
            .collect::<onyxia_ir::Result<_>>()?;
        // Cold pass (includes pipeline compilation) …
        let outs = session.run(&dev).await?;
        let (_, logits) = outs
            .iter()
            .find(|(n, _)| n == "logits")
            .ok_or_else(|| onyxia_ir::Error::Runtime("no 'logits' output".into()))?;
        let result = session.download(logits).await?;
        eprintln!("GPU forward pass (cold) in {:.2?}", started.elapsed());
        // … then a warm pass: the number that matters for decode.
        let warm = std::time::Instant::now();
        let outs = session.run(&dev).await?;
        let (_, logits) = outs
            .iter()
            .find(|(n, _)| n == "logits")
            .ok_or_else(|| onyxia_ir::Error::Runtime("no 'logits' output".into()))?;
        let _ = session.download(logits).await?;
        eprintln!("GPU forward pass (warm) in {:.2?}", warm.elapsed());
        Ok::<_, onyxia_ir::Error>(result)
    })?;
    let gpu_last = last_position(&logits_gpu, seq)?;
    print_top5("gpu", &gpu_last);

    if skip_ref {
        return Ok(());
    }

    eprintln!("running the same pass on the reference interpreter (slow) …");
    let started = std::time::Instant::now();
    let ref_outs = onyxia_backend_ref::run_once(module, &input_refs)?;
    eprintln!("reference pass in {:.2?}", started.elapsed());
    let (_, logits_ref) = ref_outs
        .iter()
        .find(|(n, _)| n == "logits")
        .ok_or("no 'logits' in reference outputs")?;
    let ref_last = last_position(logits_ref, seq)?;
    print_top5("ref", &ref_last);

    let mut max_abs = 0f32;
    for (a, b) in gpu_last.iter().zip(&ref_last) {
        max_abs = max_abs.max((a - b).abs());
    }
    let argmax = |v: &[f32]| {
        v.iter()
            .enumerate()
            .max_by(|a, b| a.1.total_cmp(b.1))
            .map(|(i, _)| i)
            .unwrap()
    };
    println!("max |Δlogit| (last position): {max_abs}");
    println!(
        "argmax: gpu={} ref={} {}",
        argmax(&gpu_last),
        argmax(&ref_last),
        if argmax(&gpu_last) == argmax(&ref_last) {
            "✓ MATCH"
        } else {
            "✗ MISMATCH"
        }
    );
    Ok(())
}

/// Build plausible prefill inputs for every module input, from its declared
/// symbolic shape bound with S=seq, past=0.
fn build_inputs(module: &Module, seq: usize) -> Result<Vec<(String, Tensor)>, onyxia_ir::Error> {
    let mut bindings = onyxia_ir::Bindings::new();
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
            let dims = ty.shape.eval(&bindings).map_err(|e| {
                onyxia_ir::Error::Binding(format!(
                    "input '{name}': cannot resolve {} — bind more symbols in \
                     forward-check ({e})",
                    ty.shape
                ))
            })?;
            let numel: usize = dims.iter().product();
            let tensor = match (name.as_str(), ty.dtype) {
                ("input_ids", DataType::I64) => {
                    // Arbitrary but fixed token ids (2 = Gemma BOS).
                    let ids: Vec<i64> = (0..numel)
                        .map(|i| if i == 0 { 2 } else { 651 + i as i64 })
                        .collect();
                    Tensor::from_i64(&ids, &dims)?
                }
                ("position_ids", DataType::I64) => {
                    Tensor::from_i64(&(0..numel as i64).collect::<Vec<_>>(), &dims)?
                }
                (_, DataType::I64) => Tensor::from_i64(&vec![1; numel], &dims)?, // masks
                (_, DataType::F32) => Tensor::from_f32(&vec![0.0; numel], &dims)?,
                (_, dt) => {
                    return Err(onyxia_ir::Error::Unsupported(format!(
                        "input '{name}' dtype {dt} in forward-check"
                    )));
                }
            };
            Ok((name.clone(), tensor))
        })
        .collect()
}

/// Logits for the last sequence position: `[1, S, V] → [V]`.
fn last_position(logits: &Tensor, seq: usize) -> Result<Vec<f32>, onyxia_ir::Error> {
    let all = logits.to_f32()?;
    let vocab = logits.shape()[logits.shape().len() - 1];
    Ok(all[(seq - 1) * vocab..seq * vocab].to_vec())
}

fn print_top5(label: &str, logits: &[f32]) {
    let mut idx: Vec<usize> = (0..logits.len()).collect();
    idx.sort_by(|&a, &b| logits[b].total_cmp(&logits[a]));
    let top: Vec<String> = idx[..5]
        .iter()
        .map(|&i| format!("{i}:{:.3}", logits[i]))
        .collect();
    println!("{label} top-5 logits: {}", top.join("  "));
}
