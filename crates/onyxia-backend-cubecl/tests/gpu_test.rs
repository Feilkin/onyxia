//! Differential tests: every module runs on the CubeCL backend and the
//! reference interpreter; results must agree within the pinned tolerances
//! (atol 1e-4, rtol 1e-3). Mirrors the wgpu backend's suite — same modules,
//! different backend — plus composite tests that exercise legalization
//! (this backend has NO composite kernels; everything decomposes).
//!
//! All tests require a GPU and are `#[ignore]`d for plain CI runs — use
//! `cargo nextest run --run-ignored=all` (or `just test-all`).

use onyxia_backend_cubecl::CubeclBackend;
use onyxia_ir::interp::Tensor;
use onyxia_ir::{
    AttrValue, Attrs, Backend, DataType, DimExpr, GraphBuilder, Module, ReduceOp, Session,
    SymbolicShape, TensorType, UnaryOp,
};

const ATOL: f32 = 1e-4;
const RTOL: f32 = 1e-3;

/// Run on the reference interpreter and the CubeCL backend; compare.
fn diff_test(module: Module, inputs: Vec<(&str, Tensor)>) {
    let expect = onyxia_backend_ref::run_once(module.clone(), &inputs).unwrap();

    let got: Vec<(String, Tensor)> = pollster::block_on(async {
        let backend = CubeclBackend::new();
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
                        "output '{en}'[{i}]: interp {a} vs cubecl {b}"
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
    let x = b.input("x", TensorType::of(DataType::F32, &[2, 3, 4]));
    let y = b.input("y", TensorType::of(DataType::F32, &[4]));
    let s = b.add(x, y).unwrap();
    let e = b.unary(UnaryOp::Exp, s).unwrap();
    let t = b.unary(UnaryOp::Tanh, e).unwrap();
    b.output("out", t);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![
            (
                "x",
                Tensor::from_f32(&f32s(24, |i| (i as f32) * 0.1 - 1.0), &[2, 3, 4]).unwrap(),
            ),
            (
                "y",
                Tensor::from_f32(&[0.5, -0.5, 1.0, 0.0], &[4]).unwrap(),
            ),
        ],
    );
}

#[test]
#[ignore = "requires GPU"]
fn matmul_batched_transposed() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[2, 3, 4]));
    let w = b.input("w", TensorType::of(DataType::F32, &[2, 5, 4]));
    let y = b
        .prim(
            onyxia_ir::Prim::MatMul {
                trans_a: false,
                trans_b: true,
            },
            &[x, w],
        )
        .unwrap(); // [2, 3, 5]
    b.output("out", y);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![
            (
                "x",
                Tensor::from_f32(&f32s(24, |i| (i as f32) * 0.3 - 2.0), &[2, 3, 4]).unwrap(),
            ),
            (
                "w",
                Tensor::from_f32(&f32s(40, |i| 1.0 - (i as f32) * 0.1), &[2, 5, 4]).unwrap(),
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
    let max = b.reduce(ReduceOp::Max, x, &[0, 2], false).unwrap();
    b.output("mean", mean);
    b.output("max", max);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![(
            "x",
            Tensor::from_f32(&f32s(24, |i| ((i * 7) % 13) as f32 - 6.0), &[2, 3, 4]).unwrap(),
        )],
    );
}

/// Softmax + RMS-norm run as composites on the wgpu backend; here they MUST
/// legalize through their decompositions — this backend's core claim.
#[test]
#[ignore = "requires GPU"]
fn softmax_and_rmsnorm_decompose() {
    let mut b = GraphBuilder::new();
    let x = b.input("x", TensorType::of(DataType::F32, &[2, 8]));
    let w = b.input("w", TensorType::of(DataType::F32, &[8]));
    let sm = b
        .composite(
            "Softmax",
            Attrs::new().with("axis", AttrValue::Int(1)),
            &[x],
            vec![TensorType::of(DataType::F32, &[2, 8])],
        )
        .unwrap()
        .remove(0);
    let rms = b
        .composite(
            "SimplifiedLayerNormalization",
            Attrs::new().with("epsilon", AttrValue::Float(1e-6)),
            &[sm, w],
            vec![TensorType::of(DataType::F32, &[2, 8])],
        )
        .unwrap()
        .remove(0);
    b.output("out", rms);
    let m = b.finish().unwrap();
    diff_test(
        m,
        vec![
            (
                "x",
                Tensor::from_f32(&f32s(16, |i| (i as f32) * 0.7 - 5.0), &[2, 8]).unwrap(),
            ),
            (
                "w",
                Tensor::from_f32(&f32s(8, |i| 0.5 + (i as f32) * 0.2), &[8]).unwrap(),
            ),
        ],
    );
}

#[test]
#[ignore = "requires GPU"]
fn gather_concat_slice_i64() {
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
    b.output("out", joined);
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

/// GQA with symbolic dims, past KV, and a sliding window — the full
/// decomposition (transpose, concat, broadcast, matmul, iota, compare,
/// select, softmax-decomposition) on the CubeCL backend.
#[test]
#[ignore = "requires GPU"]
fn gqa_decomposes_with_symbolic_dims() {
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
                    &f32s(s_c * kv_hidden, |i| (i as f32 * 0.23).cos()),
                    &[1, s_c, kv_hidden],
                )
                .unwrap(),
            ),
            (
                "v",
                Tensor::from_f32(
                    &f32s(s_c * kv_hidden, |i| (i as f32) * 0.3 - 1.5),
                    &[1, s_c, kv_hidden],
                )
                .unwrap(),
            ),
            (
                "pk",
                Tensor::from_f32(
                    &f32s(t_c * kv_hidden, |i| (i as f32 * 0.41).sin()),
                    &[1, kv as usize, t_c, d],
                )
                .unwrap(),
            ),
            (
                "pv",
                Tensor::from_f32(
                    &f32s(t_c * kv_hidden, |i| 1.0 - (i as f32) * 0.2),
                    &[1, kv as usize, t_c, d],
                )
                .unwrap(),
            ),
        ],
    );
}
