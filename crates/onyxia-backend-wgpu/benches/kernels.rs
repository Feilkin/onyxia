//! Kernel microbenchmarks at LLM-shaped sizes (requires a GPU).
//!
//! Each case builds a single-op IR module with the weight as a module
//! constant (resident on device, like a real model), then times
//! `Session::run` + wait-idle — i.e. encode + submit + GPU execution,
//! no readback. Shapes follow Gemma 3 270m (hidden 640, ffn 2048,
//! vocab 262144); throughput is reported as weight-bytes moved, so
//! matmul rows read directly as effective GB/s against the card's
//! memory bandwidth.
//!
//! Run with `cargo bench -p onyxia-backend-wgpu`. Compare runs with
//! `critcmp` or criterion's built-in baseline diffing.

use criterion::{Criterion, Throughput, criterion_group, criterion_main};
use onyxia_backend_wgpu::{GpuContext, WgpuBackend, WgpuSession};
use onyxia_ir::interp::Tensor;
use onyxia_ir::{Backend, DataType, GraphBuilder, Module, Prim, Session, TensorType};

const HIDDEN: usize = 640;
const FFN: usize = 2048;
const VOCAB: usize = 262144;

/// A prepared session plus uploaded inputs, ready to run repeatedly.
struct Case {
    session: WgpuSession,
    inputs: Vec<(String, onyxia_backend_wgpu::GpuTensor)>,
}

impl Case {
    fn new(ctx: &GpuContext, module: Module, inputs: Vec<(&str, Tensor)>) -> Self {
        let backend = WgpuBackend::new(GpuContext {
            device: ctx.device.clone(),
            queue: ctx.queue.clone(),
            adapter_info: ctx.adapter_info.clone(),
            use_immediates: ctx.use_immediates,
        });
        let mut session = backend.prepare(module).expect("prepare");
        let inputs = inputs
            .into_iter()
            .map(|(n, t)| (n.to_string(), session.upload(&t).expect("upload")))
            .collect();
        Self { session, inputs }
    }

    /// One timed iteration: full dispatch stream + GPU completion.
    fn run(&mut self) {
        let refs: Vec<(&str, _)> = self
            .inputs
            .iter()
            .map(|(n, t)| (n.as_str(), t.clone()))
            .collect();
        pollster::block_on(self.session.run(&refs)).expect("run");
        self.session.wait_idle().expect("wait");
    }
}

fn f32s(n: usize) -> Vec<f32> {
    (0..n).map(|i| (i % 977) as f32 * 1e-3 - 0.5).collect()
}

/// `[m,k] × [k,n]` (or `[n,k]` with `trans_b`) with the rhs as a module
/// constant, mirroring an activation × resident-weight matmul.
fn matmul_case(ctx: &GpuContext, m: usize, k: usize, n: usize, trans_b: bool) -> Case {
    let mut b = GraphBuilder::new();
    let a = b.input("a", TensorType::of(DataType::F32, &[m as u64, k as u64]));
    let w_dims = if trans_b { [n as u64, k as u64] } else { [k as u64, n as u64] };
    let w = b.const_f32(&f32s(k * n), &w_dims).unwrap();
    let out = b
        .prim(
            Prim::MatMul {
                trans_a: false,
                trans_b,
            },
            &[a, w],
        )
        .unwrap();
    b.output("out", out);
    Case::new(
        ctx,
        b.finish().unwrap(),
        vec![("a", Tensor::from_f32(&f32s(m * k), &[m, k]).unwrap())],
    )
}

/// Transpose of a resident `[rows, cols]` constant (the `lm_head`
/// pattern the lowerer currently emits every step).
fn transpose_case(ctx: &GpuContext, rows: usize, cols: usize) -> Case {
    let mut b = GraphBuilder::new();
    let w = b
        .const_f32(&f32s(rows * cols), &[rows as u64, cols as u64])
        .unwrap();
    let out = b.transpose(w, &[1, 0]).unwrap();
    b.output("out", out);
    Case::new(ctx, b.finish().unwrap(), vec![])
}

/// A chain of `depth` tiny elementwise adds on `[1, HIDDEN]` — measures
/// per-dispatch overhead (encode + bind + submit + launch), which is what
/// a decode step pays ~385 times.
fn dispatch_chain_case(ctx: &GpuContext, depth: usize) -> Case {
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[1, HIDDEN as u64]));
    let c = b.const_f32(&f32s(HIDDEN), &[1, HIDDEN as u64]).unwrap();
    let mut v = x;
    for _ in 0..depth {
        v = b.add(v, c).unwrap();
    }
    b.output("out", v);
    Case::new(
        ctx,
        b.finish().unwrap(),
        vec![("x", Tensor::from_f32(&f32s(HIDDEN), &[1, HIDDEN]).unwrap())],
    )
}

fn bench_kernels(c: &mut Criterion) {
    let ctx = pollster::block_on(GpuContext::new()).expect("GPU available");
    eprintln!(
        "adapter: {} ({:?})",
        ctx.adapter_info.name, ctx.adapter_info.backend
    );

    let weight_bytes = |k: usize, n: usize| (k * n * 4) as u64;
    let mut g = c.benchmark_group("decode_matmul");
    g.sample_size(20);

    // lm_head: the dominant weight. Both layouts, to compare the
    // transpose-then-matmul status quo against a trans_b matmul.
    g.throughput(Throughput::Bytes(weight_bytes(HIDDEN, VOCAB)));
    let mut case = matmul_case(&ctx, 1, HIDDEN, VOCAB, false);
    g.bench_function("lm_head_1x640x262144", |b| b.iter(|| case.run()));
    let mut case = matmul_case(&ctx, 1, HIDDEN, VOCAB, true);
    g.bench_function("lm_head_1x640x262144_transb", |b| b.iter(|| case.run()));

    // MLP projections.
    g.throughput(Throughput::Bytes(weight_bytes(HIDDEN, FFN)));
    let mut case = matmul_case(&ctx, 1, HIDDEN, FFN, false);
    g.bench_function("gate_1x640x2048", |b| b.iter(|| case.run()));
    g.throughput(Throughput::Bytes(weight_bytes(FFN, HIDDEN)));
    let mut case = matmul_case(&ctx, 1, FFN, HIDDEN, false);
    g.bench_function("down_1x2048x640", |b| b.iter(|| case.run()));
    g.finish();

    let mut g = c.benchmark_group("prefill_matmul");
    g.sample_size(10);
    g.throughput(Throughput::Bytes(weight_bytes(HIDDEN, VOCAB)));
    let mut case = matmul_case(&ctx, 64, HIDDEN, VOCAB, false);
    g.bench_function("lm_head_64x640x262144", |b| b.iter(|| case.run()));
    g.throughput(Throughput::Bytes(weight_bytes(FFN, HIDDEN)));
    let mut case = matmul_case(&ctx, 64, FFN, HIDDEN, false);
    g.bench_function("down_64x2048x640", |b| b.iter(|| case.run()));
    g.finish();

    let mut g = c.benchmark_group("data_movement");
    g.sample_size(10);
    g.throughput(Throughput::Bytes(weight_bytes(VOCAB, HIDDEN)));
    let mut case = transpose_case(&ctx, VOCAB, HIDDEN);
    g.bench_function("transpose_262144x640", |b| b.iter(|| case.run()));
    g.finish();

    let mut g = c.benchmark_group("dispatch_overhead");
    g.sample_size(30);
    // Per-dispatch cost = chain(400) time / 400; compare against
    // chain(1) to separate fixed submit cost from per-dispatch cost.
    let mut case = dispatch_chain_case(&ctx, 1);
    g.bench_function("chain_1", |b| b.iter(|| case.run()));
    let mut case = dispatch_chain_case(&ctx, 400);
    g.bench_function("chain_400", |b| b.iter(|| case.run()));
    g.finish();
}

criterion_group!(benches, bench_kernels);
criterion_main!(benches);
