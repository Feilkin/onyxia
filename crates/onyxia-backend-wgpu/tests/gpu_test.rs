//! Differential tests: every module runs on both the wgpu backend and the
//! reference interpreter; results must agree within the pinned tolerances
//! (atol 1e-4, rtol 1e-3).
//!
//! All tests require a GPU and are `#[ignore]`d for plain CI runs — use
//! `cargo nextest run --run-ignored=all` (or `just test-all`).

use onyxia_backend_wgpu::{GpuContext, WgpuBackend};
use onyxia_ir::interp::Tensor;
use onyxia_ir::{
    AttrValue, Attrs, Backend, DataType, DimExpr, GraphBuilder, Module, ReduceOp, Session,
    SymbolicShape, TensorType, UnaryOp,
};

const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-3;

/// Run on both backends and compare all outputs.
fn diff_test(module: Module, inputs: Vec<(&str, Tensor)>) {
    let expect = onyxia_backend_ref::run_once(module.clone(), &inputs).unwrap();

    let got: Vec<(String, Tensor)> = pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("GPU available");
        let backend = WgpuBackend::new(ctx);
        let mut session = backend.prepare(module).expect("prepare");
        let dev_inputs: Vec<(&str, _)> = inputs
            .iter()
            .map(|(n, t)| (*n, session.upload(t).expect("upload")))
            .collect();
        let outs = session.run(&dev_inputs).await.expect("run");
        let mut host = Vec::new();
        for (n, t) in outs {
            host.push((n, session.download(&t).await.expect("download")));
        }
        host
    });

    assert_eq!(expect.len(), got.len());
    for ((en, et), (gn, gt)) in expect.iter().zip(&got) {
        assert_eq!(en, gn);
        assert_eq!(et.shape(), gt.shape(), "output '{en}' shape");
        assert_eq!(et.dtype(), gt.dtype(), "output '{en}' dtype");
        match et.dtype() {
            DataType::F32 => {
                let (e, g) = (et.to_f32().unwrap(), gt.to_f32().unwrap());
                for (i, (a, b)) in e.iter().zip(&g).enumerate() {
                    assert!(
                        (a - b).abs() <= ATOL + RTOL * a.abs(),
                        "output '{en}'[{i}]: interp {a} vs gpu {b}"
                    );
                }
            }
            DataType::I64 => {
                assert_eq!(et.to_i64().unwrap(), gt.to_i64().unwrap(), "output '{en}'")
            }
            DataType::Bool => assert_eq!(
                et.to_bool().unwrap(),
                gt.to_bool().unwrap(),
                "output '{en}'"
            ),
            other => panic!("unexpected output dtype {other}"),
        }
    }
}

fn f32s(n: usize, f: impl Fn(usize) -> f32) -> Vec<f32> {
    (0..n).map(f).collect()
}

#[test]
#[ignore = "requires GPU"]
fn elementwise_broadcast_and_unary() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[2, 3]));
    let y = b.input("y", TensorType::of(DataType::F32, &[3]));
    let sum = b.add(x, y).unwrap();
    let t = b.unary(UnaryOp::Tanh, sum).unwrap();
    let e = b.unary(UnaryOp::Erf, t).unwrap();
    b.output("out", e);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![
            (
                "x",
                Tensor::from_f32(&f32s(6, |i| i as f32 * 0.3 - 1.0), &[2, 3]).unwrap(),
            ),
            ("y", Tensor::from_f32(&[0.5, -0.5, 2.0], &[3]).unwrap()),
        ],
    );
}

#[test]
#[ignore = "requires GPU"]
fn matmul_batched_transposed() {
    let mut b = GraphBuilder::new();
    let a = b.input("a", TensorType::of(DataType::F32, &[2, 3, 4]));
    let w = b.input("w", TensorType::of(DataType::F32, &[5, 4]));
    // a[2,3,4] × w[5,4]ᵀ → [2,3,5], batched with a broadcast rhs.
    let out = b
        .prim(
            onyxia_ir::Prim::MatMul {
                trans_a: false,
                trans_b: true,
            },
            &[a, w],
        )
        .unwrap();
    b.output("out", out);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![
            (
                "a",
                Tensor::from_f32(&f32s(24, |i| (i as f32).sin()), &[2, 3, 4]).unwrap(),
            ),
            (
                "w",
                Tensor::from_f32(&f32s(20, |i| (i as f32) * 0.05 - 0.4), &[5, 4]).unwrap(),
            ),
        ],
    );
}

#[test]
#[ignore = "requires GPU"]
fn reduce_mean_and_max() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[2, 3, 4]));
    let mean = b.reduce(ReduceOp::Mean, x, &[2], true).unwrap();
    let mx = b.reduce(ReduceOp::Max, x, &[0, 1], false).unwrap();
    b.output("mean", mean);
    b.output("max", mx);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![(
            "x",
            Tensor::from_f32(&f32s(24, |i| ((i * 7) % 13) as f32 - 6.0), &[2, 3, 4]).unwrap(),
        )],
    );
}

#[test]
#[ignore = "requires GPU"]
fn softmax_and_rmsnorm_composites() {
    // Composites inline during prepare; this exercises the whole
    // decomposition on GPU.
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[2, 8]));
    let w = b.input("w", TensorType::of(DataType::F32, &[8]));
    let normed = b
        .composite(
            "SimplifiedLayerNormalization",
            Attrs::new().with("epsilon", AttrValue::Float(1e-6)),
            &[x, w],
            vec![TensorType::of(DataType::F32, &[2, 8])],
        )
        .unwrap()[0];
    let soft = b
        .composite(
            "Softmax",
            Attrs::new().with("axis", AttrValue::Int(1)),
            &[normed],
            vec![TensorType::of(DataType::F32, &[2, 8])],
        )
        .unwrap()[0];
    b.output("out", soft);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![
            (
                "x",
                Tensor::from_f32(&f32s(16, |i| i as f32 * 0.7 - 5.0), &[2, 8]).unwrap(),
            ),
            (
                "w",
                Tensor::from_f32(&f32s(8, |i| 1.0 + i as f32 * 0.1), &[8]).unwrap(),
            ),
        ],
    );
}

/// The money test: GQA with symbolic S and T, past KV, and a sliding
/// window — the full decomposition (reshapes, concat, broadcast repeat,
/// iota, symbolic-start slice, compare, select, softmax, matmuls) on GPU.
#[test]
#[ignore = "requires GPU"]
fn gqa_symbolic_with_past_and_window() {
    let (h, kv, d) = (2u64, 1u64, 4usize);
    let hidden = h as usize * d;
    let kv_hidden = kv as usize * d;

    let mut b = GraphBuilder::new();
    let s = b.sym("S");
    let t = b.sym("T");
    let f = DataType::F32;
    let dim = |v: usize| DimExpr::constant(v as u64);
    let q = b.input(
        "q",
        TensorType::new(f, SymbolicShape(vec![dim(1), s.clone(), dim(hidden)])),
    );
    let k = b.input(
        "k",
        TensorType::new(f, SymbolicShape(vec![dim(1), s.clone(), dim(kv_hidden)])),
    );
    let v = b.input(
        "v",
        TensorType::new(f, SymbolicShape(vec![dim(1), s.clone(), dim(kv_hidden)])),
    );
    let pk = b.input(
        "pk",
        TensorType::new(
            f,
            SymbolicShape(vec![dim(1), DimExpr::constant(kv), t.clone(), dim(d)]),
        ),
    );
    let pv = b.input(
        "pv",
        TensorType::new(
            f,
            SymbolicShape(vec![dim(1), DimExpr::constant(kv), t.clone(), dim(d)]),
        ),
    );
    let present = SymbolicShape(vec![dim(1), DimExpr::constant(kv), t + s.clone(), dim(d)]);
    let outs = b
        .composite(
            "com.microsoft.GroupQueryAttention",
            Attrs::new()
                .with("num_heads", AttrValue::Int(h as i64))
                .with("kv_num_heads", AttrValue::Int(kv as i64))
                .with("local_window_size", AttrValue::Int(3)),
            &[q, k, v, pk, pv],
            vec![
                TensorType::new(f, SymbolicShape(vec![dim(1), s, dim(hidden)])),
                TensorType::new(f, present.clone()),
                TensorType::new(f, present),
            ],
        )
        .unwrap();
    b.output("out", outs[0]);
    b.output("present_k", outs[1]);
    b.output("present_v", outs[2]);
    let m = b.finish().unwrap();

    // S=3 new tokens over T=2 past tokens.
    let (s_c, t_c) = (3usize, 2usize);
    diff_test(
        m,
        vec![
            (
                "q",
                Tensor::from_f32(
                    &f32s(s_c * hidden, |i| (i as f32 * 0.11).sin()),
                    &[1, s_c, hidden],
                )
                .unwrap(),
            ),
            (
                "k",
                Tensor::from_f32(
                    &f32s(s_c * kv_hidden, |i| (i as f32 * 0.2).cos()),
                    &[1, s_c, kv_hidden],
                )
                .unwrap(),
            ),
            (
                "v",
                Tensor::from_f32(
                    &f32s(s_c * kv_hidden, |i| i as f32 * 0.3),
                    &[1, s_c, kv_hidden],
                )
                .unwrap(),
            ),
            (
                "pk",
                Tensor::from_f32(&f32s(t_c * d, |i| 0.4 - i as f32 * 0.1), &[1, 1, t_c, d])
                    .unwrap(),
            ),
            (
                "pv",
                Tensor::from_f32(&f32s(t_c * d, |i| 5.0 + i as f32), &[1, 1, t_c, d]).unwrap(),
            ),
        ],
    );
}

#[test]
#[ignore = "requires GPU"]
fn gather_concat_slice_scatter_i64() {
    let mut b = GraphBuilder::new();
    let table = b.input("table", TensorType::of(DataType::F32, &[10, 3]));
    let ids = b.input("ids", TensorType::of(DataType::I64, &[4]));
    let rows = b.gather(table, ids, 0).unwrap(); // [4, 3]
    let first2 = b
        .slice(
            rows,
            vec![onyxia_ir::SliceSpec {
                axis: 0,
                start: DimExpr::constant(0),
                end: DimExpr::constant(2),
                step: 1,
            }],
        )
        .unwrap(); // [2, 3]
    let joined = b.concat(&[rows, first2], 0).unwrap(); // [6, 3]
    // Scatter row 5 <- row 0 values.
    let idx = b.const_i64(&[5], &[1, 1]).unwrap();
    let upd = b.const_f32(&[9.0, 9.5, 10.0], &[1, 3]).unwrap();
    let scat = b
        .prim(onyxia_ir::Prim::Scatter, &[joined, idx, upd])
        .unwrap();
    b.output("out", scat);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![
            (
                "table",
                Tensor::from_f32(&f32s(30, |i| i as f32), &[10, 3]).unwrap(),
            ),
            ("ids", Tensor::from_i64(&[7, 0, -1, 3], &[4]).unwrap()),
        ],
    );
}

#[test]
#[ignore = "requires GPU"]
fn buffer_pool_reuses_on_second_run() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[64]));
    let a = b.add(x, x).unwrap();
    let c = b.mul(a, a).unwrap();
    let d = b.sub(c, x).unwrap();
    b.output("out", d);
    let m = b.finish().unwrap();

    pollster::block_on(async {
        let ctx = GpuContext::new().await.expect("GPU available");
        let backend = WgpuBackend::new(ctx);
        let mut session = backend.prepare(m).expect("prepare");
        let input = Tensor::from_f32(&f32s(64, |i| i as f32), &[64]).unwrap();
        let dev = session.upload(&input).unwrap();
        let _ = session.run(&[("x", dev.clone())]).await.unwrap();
        let (alloc1, _) = session.pool_stats();
        let _ = session.run(&[("x", dev)]).await.unwrap();
        let (alloc2, reuses) = session.pool_stats();
        assert!(reuses > 0, "second run must reuse pooled buffers");
        // Output buffers leave the session with the caller's handle (they
        // are dropped, not pooled), so at most one fresh allocation per run
        // replaces the departed output buffer.
        assert!(
            alloc2 <= alloc1 + 1,
            "second run over-allocated: {alloc1} -> {alloc2}"
        );
    });
}
