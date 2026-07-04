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

/// Run on both backends and compare all outputs. The GPU side runs twice:
/// with immediates (native fast path) and with the storage-buffer params
/// fallback (the web path, where WebGPU has no push constants) — so every
/// differential test covers both dispatch modes.
fn diff_test(module: Module, inputs: Vec<(&str, Tensor)>) {
    let expect = onyxia_backend_ref::run_once(module.clone(), &inputs).unwrap();

    for immediates in [true, false] {
        let got: Vec<(String, Tensor)> = pollster::block_on(async {
            let ctx = GpuContext::new_with(immediates)
                .await
                .expect("GPU available");
            let backend = WgpuBackend::new(ctx);
            let mut session = backend.prepare(module.clone()).expect("prepare");
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

        let mode = if immediates { "immediates" } else { "fallback" };
        assert_eq!(expect.len(), got.len());
        for ((en, et), (gn, gt)) in expect.iter().zip(&got) {
            assert_eq!(en, gn);
            assert_eq!(et.shape(), gt.shape(), "[{mode}] output '{en}' shape");
            assert_eq!(et.dtype(), gt.dtype(), "[{mode}] output '{en}' dtype");
            match et.dtype() {
                DataType::F32 => {
                    let (e, g) = (et.to_f32().unwrap(), gt.to_f32().unwrap());
                    for (i, (a, b)) in e.iter().zip(&g).enumerate() {
                        assert!(
                            (a - b).abs() <= ATOL + RTOL * a.abs(),
                            "[{mode}] output '{en}'[{i}]: interp {a} vs gpu {b}"
                        );
                    }
                }
                DataType::I64 => assert_eq!(
                    et.to_i64().unwrap(),
                    gt.to_i64().unwrap(),
                    "[{mode}] output '{en}'"
                ),
                DataType::Bool => assert_eq!(
                    et.to_bool().unwrap(),
                    gt.to_bool().unwrap(),
                    "[{mode}] output '{en}'"
                ),
                other => panic!("unexpected output dtype {other}"),
            }
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

/// The M=1 matvec fast path, across layouts, split-K depths, and sizes
/// that don't divide the 64-column tile or the K slices evenly.
#[test]
#[ignore = "requires GPU"]
fn matvec_m1_fast_path() {
    // (k, n, trans_b): decode-shaped projections both ways, small-N
    // cases that force ks > 1, awkward remainders, and both sides of
    // the vec4 alignment gates (K % 4 for [N,K], N % 4 for [K,N]).
    for (k, n, trans_b) in [
        (2048usize, 640usize, false), // vec4 [K,N] with ks>1
        (640, 2048, false),           // gate/up
        (640, 2048, true),            // vec4 [N,K], ks=1
        (4096, 96, true),             // vec4 [N,K] with ks>1
        (4096, 66, true),             // vec4 [N,K], row-group tail
        (2048, 644, false),           // vec4 [K,N], column-tile tail
        (2048, 2, true),              // vec4 [N,K], n smaller than a group
        (130, 100, false),            // scalar [K,N] (N % 4 ≠ 0)
        (1, 1, false),                // degenerate
        (257, 65, true),              // scalar [N,K] (K % 4 ≠ 0)
    ] {
        let mut b = GraphBuilder::new();
        let a = b.input("a", TensorType::of(DataType::F32, &[1, k as u64]));
        let w_dims = if trans_b {
            [n as u64, k as u64]
        } else {
            [k as u64, n as u64]
        };
        let w = b.input("w", TensorType::of(DataType::F32, &w_dims));
        let out = b
            .prim(
                onyxia_ir::Prim::MatMul {
                    trans_a: false,
                    trans_b,
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
                    Tensor::from_f32(&f32s(k, |i| (i as f32 * 0.7).sin()), &[1, k]).unwrap(),
                ),
                (
                    "w",
                    Tensor::from_f32(
                        &f32s(k * n, |i| ((i % 601) as f32) * 1e-3 - 0.3),
                        &[w_dims[0] as usize, w_dims[1] as usize],
                    )
                    .unwrap(),
                ),
            ],
        );
    }
}

/// The tiled m>1 path: all four layout variants, sizes that straddle the
/// 16×16 tiles, batched lhs with a broadcast rank-2 rhs.
#[test]
#[ignore = "requires GPU"]
fn matmul_tiled_all_layouts() {
    for (trans_a, trans_b) in [(false, false), (false, true), (true, false), (true, true)] {
        let (m, k, n) = (33usize, 70usize, 45usize);
        let a_dims = if trans_a { [k, m] } else { [m, k] };
        let b_dims = if trans_b { [n, k] } else { [k, n] };
        let mut bld = GraphBuilder::new();
        let a = bld.input(
            "a",
            TensorType::of(DataType::F32, &[3, a_dims[0] as u64, a_dims[1] as u64]),
        );
        let w = bld.input(
            "w",
            TensorType::of(DataType::F32, &[b_dims[0] as u64, b_dims[1] as u64]),
        );
        let out = bld
            .prim(onyxia_ir::Prim::MatMul { trans_a, trans_b }, &[a, w])
            .unwrap();
        bld.output("out", out);
        let module = bld.finish().unwrap();
        diff_test(
            module,
            vec![
                (
                    "a",
                    Tensor::from_f32(
                        &f32s(3 * m * k, |i| (i as f32 * 0.31).sin()),
                        &[3, a_dims[0], a_dims[1]],
                    )
                    .unwrap(),
                ),
                (
                    "w",
                    Tensor::from_f32(
                        &f32s(k * n, |i| ((i % 401) as f32) * 2e-3 - 0.4),
                        &[b_dims[0], b_dims[1]],
                    )
                    .unwrap(),
                ),
            ],
        );
    }
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

/// Fused rotary embedding vs the interpreter: decode-like offset
/// positions, inferred head count, and a partial rotary width with a
/// pass-through tail.
#[test]
#[ignore = "requires GPU"]
fn rotary_fused_matches_interpreter() {
    // (hidden, num_heads attr, cache half-width): full-width rotation
    // with inferred heads, and a partial rotation (d=8, R=4) with the
    // head count pinned by the attribute.
    for (hidden, heads_attr, half) in [(8usize, 0i64, 2usize), (16, 2, 2)] {
        let (s, max_seq) = (3usize, 16usize);
        let mut b = GraphBuilder::new();
        let x = b.input("x", TensorType::of(DataType::F32, &[1, s as u64, hidden as u64]));
        let pos = b.input("pos", TensorType::of(DataType::I64, &[1, s as u64]));
        let cos = b.input(
            "cos",
            TensorType::of(DataType::F32, &[max_seq as u64, half as u64]),
        );
        let sin = b.input(
            "sin",
            TensorType::of(DataType::F32, &[max_seq as u64, half as u64]),
        );
        let out = b
            .composite(
                "com.microsoft.RotaryEmbedding",
                Attrs::new().with("num_heads", AttrValue::Int(heads_attr)),
                &[x, pos, cos, sin],
                vec![TensorType::of(DataType::F32, &[1, s as u64, hidden as u64])],
            )
            .unwrap()[0];
        b.output("out", out);
        let m = b.finish().unwrap();
        diff_test(
            m,
            vec![
                (
                    "x",
                    Tensor::from_f32(&f32s(s * hidden, |i| (i as f32 * 0.23).sin()), &[1, s, hidden])
                        .unwrap(),
                ),
                ("pos", Tensor::from_i64(&[5, 6, 7], &[1, s]).unwrap()),
                (
                    "cos",
                    Tensor::from_f32(&f32s(max_seq * half, |i| (i as f32 * 0.05).cos()), &[max_seq, half])
                        .unwrap(),
                ),
                (
                    "sin",
                    Tensor::from_f32(&f32s(max_seq * half, |i| (i as f32 * 0.05).sin()), &[max_seq, half])
                        .unwrap(),
                ),
            ],
        );
    }
}

/// Fused rotary + GQA against their decompositions on the same device —
/// the decomposition is the specification. Batch of 2, grouped heads
/// (H=4 over KV=2), past KV, and a sliding window, chained the way a
/// transformer layer uses them.
#[test]
#[ignore = "requires GPU"]
fn gqa_rotary_fused_match_decompositions() {
    let (bsz, s, h, kv, d, past) = (2usize, 3usize, 4usize, 2usize, 8usize, 5usize);
    let (hidden, kv_hidden, total, max_seq, half) = (h * d, kv * d, past + s, 16usize, d / 2);

    let mut bld = GraphBuilder::new();
    let dims = |v: &[usize]| v.iter().map(|&x| x as u64).collect::<Vec<_>>();
    let q = bld.input("q", TensorType::of(DataType::F32, &dims(&[bsz, s, hidden])));
    let k = bld.input("k", TensorType::of(DataType::F32, &dims(&[bsz, s, kv_hidden])));
    let v = bld.input("v", TensorType::of(DataType::F32, &dims(&[bsz, s, kv_hidden])));
    let pk = bld.input(
        "pk",
        TensorType::of(DataType::F32, &dims(&[bsz, kv, past, d])),
    );
    let pv = bld.input(
        "pv",
        TensorType::of(DataType::F32, &dims(&[bsz, kv, past, d])),
    );
    let pos = bld.input("pos", TensorType::of(DataType::I64, &dims(&[bsz, s])));
    let cos = bld.input(
        "cos",
        TensorType::of(DataType::F32, &dims(&[max_seq, half])),
    );
    let sin = bld.input(
        "sin",
        TensorType::of(DataType::F32, &dims(&[max_seq, half])),
    );

    let rotary = |bld: &mut GraphBuilder, x, width: usize| {
        bld.composite(
            "com.microsoft.RotaryEmbedding",
            Attrs::new(),
            &[x, pos, cos, sin],
            vec![TensorType::of(DataType::F32, &dims(&[bsz, s, width]))],
        )
        .unwrap()[0]
    };
    let rq = rotary(&mut bld, q, hidden);
    let rk = rotary(&mut bld, k, kv_hidden);
    let present_ty = TensorType::of(DataType::F32, &dims(&[bsz, kv, total, d]));
    let outs = bld
        .composite(
            "com.microsoft.GroupQueryAttention",
            Attrs::new()
                .with("num_heads", AttrValue::Int(h as i64))
                .with("kv_num_heads", AttrValue::Int(kv as i64))
                .with("local_window_size", AttrValue::Int(4)),
            &[rq, rk, v, pk, pv],
            vec![
                TensorType::of(DataType::F32, &dims(&[bsz, s, hidden])),
                present_ty.clone(),
                present_ty,
            ],
        )
        .unwrap();
    bld.output("out", outs[0]);
    bld.output("present_k", outs[1]);
    bld.output("present_v", outs[2]);
    let module = bld.finish().unwrap();

    let inputs: Vec<(&str, Tensor)> = vec![
        (
            "q",
            Tensor::from_f32(&f32s(bsz * s * hidden, |i| (i as f32 * 0.11).sin()), &[bsz, s, hidden])
                .unwrap(),
        ),
        (
            "k",
            Tensor::from_f32(
                &f32s(bsz * s * kv_hidden, |i| (i as f32 * 0.2).cos()),
                &[bsz, s, kv_hidden],
            )
            .unwrap(),
        ),
        (
            "v",
            Tensor::from_f32(
                &f32s(bsz * s * kv_hidden, |i| (i as f32 * 0.017).sin() * 2.0),
                &[bsz, s, kv_hidden],
            )
            .unwrap(),
        ),
        (
            "pk",
            Tensor::from_f32(
                &f32s(bsz * kv * past * d, |i| 0.4 - (i as f32 * 0.31).sin() * 0.5),
                &[bsz, kv, past, d],
            )
            .unwrap(),
        ),
        (
            "pv",
            Tensor::from_f32(
                &f32s(bsz * kv * past * d, |i| (i as f32 * 0.07).cos()),
                &[bsz, kv, past, d],
            )
            .unwrap(),
        ),
        (
            "pos",
            Tensor::from_i64(&[5, 6, 7, 5, 6, 7], &[bsz, s]).unwrap(),
        ),
        (
            "cos",
            Tensor::from_f32(&f32s(max_seq * half, |i| (i as f32 * 0.09).cos()), &[max_seq, half])
                .unwrap(),
        ),
        (
            "sin",
            Tensor::from_f32(&f32s(max_seq * half, |i| (i as f32 * 0.09).sin()), &[max_seq, half])
                .unwrap(),
        ),
    ];

    let run = |backend: WgpuBackend, module: Module| -> Vec<(String, Vec<f32>)> {
        pollster::block_on(async {
            let mut session = backend.prepare(module).unwrap();
            let dev: Vec<(&str, _)> = inputs
                .iter()
                .map(|(n, t)| (*n, session.upload(t).unwrap()))
                .collect();
            let outs = session.run(&dev).await.unwrap();
            let mut host = Vec::new();
            for (n, t) in outs {
                host.push((n, session.download(&t).await.unwrap().to_f32().unwrap()));
            }
            host
        })
    };

    let fused = run(
        WgpuBackend::new(pollster::block_on(GpuContext::new()).unwrap()),
        module.clone(),
    );
    let decomposed = run(
        WgpuBackend::without_fused_kernels(pollster::block_on(GpuContext::new()).unwrap()),
        module,
    );
    for ((name, f), (_, dec)) in fused.iter().zip(&decomposed) {
        for (i, (a, b)) in f.iter().zip(dec).enumerate() {
            assert!(
                (a - b).abs() <= ATOL + RTOL * b.abs(),
                "'{name}'[{i}]: fused {a} vs decomposed {b}"
            );
        }
    }
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

/// Fused kernels vs their decompositions, on the same device: build a
/// module exercising Softmax, RMS-norm, and Gelu (both forms), prepare it
/// once with the fused registry and once without, and require agreement.
#[test]
#[ignore = "requires GPU"]
fn fused_kernels_match_decompositions() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[3, 16]));
    let w = b.input("w", TensorType::of(DataType::F32, &[16]));
    let ty = TensorType::of(DataType::F32, &[3, 16]);
    let normed = b
        .composite(
            "SimplifiedLayerNormalization",
            Attrs::new().with("epsilon", AttrValue::Float(1e-6)),
            &[x, w],
            vec![ty.clone()],
        )
        .unwrap()[0];
    let g1 = b
        .composite("Gelu", Attrs::new(), &[normed], vec![ty.clone()])
        .unwrap()[0];
    let g2 = b
        .composite(
            "Gelu",
            Attrs::new().with("approximate", AttrValue::Str("tanh".into())),
            &[g1],
            vec![ty.clone()],
        )
        .unwrap()[0];
    let soft = b
        .composite(
            "Softmax",
            Attrs::new().with("axis", AttrValue::Int(1)),
            &[g2],
            vec![ty],
        )
        .unwrap()[0];
    b.output("out", soft);
    let module = b.finish().unwrap();

    let inputs = [
        (
            "x",
            Tensor::from_f32(&f32s(48, |i| (i as f32 * 0.37).sin() * 3.0), &[3, 16]).unwrap(),
        ),
        (
            "w",
            Tensor::from_f32(&f32s(16, |i| 0.8 + i as f32 * 0.05), &[16]).unwrap(),
        ),
    ];

    let run = |backend: WgpuBackend, module: Module| -> Vec<f32> {
        pollster::block_on(async {
            let mut session = backend.prepare(module).unwrap();
            let dev: Vec<(&str, _)> = inputs
                .iter()
                .map(|(n, t)| (*n, session.upload(t).unwrap()))
                .collect();
            let outs = session.run(&dev).await.unwrap();
            session
                .download(&outs[0].1)
                .await
                .unwrap()
                .to_f32()
                .unwrap()
        })
    };

    let fused = run(
        WgpuBackend::new(pollster::block_on(GpuContext::new()).unwrap()),
        module.clone(),
    );
    let decomposed = run(
        WgpuBackend::without_fused_kernels(pollster::block_on(GpuContext::new()).unwrap()),
        module,
    );
    for (i, (a, b)) in fused.iter().zip(&decomposed).enumerate() {
        assert!(
            (a - b).abs() <= ATOL + RTOL * b.abs(),
            "fused[{i}]={a} vs decomposed[{i}]={b}"
        );
    }
}
