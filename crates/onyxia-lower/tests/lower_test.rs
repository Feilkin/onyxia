//! Integration tests: programmatically built ONNX graphs, lowered and then
//! executed with the reference interpreter.

use onyxia_ir::interp::{Tensor, eval};
use onyxia_ir::{NodeKind, Prim, inline_composites, standard_decompositions};
use onyxia_lower::{lower, lower_with_stats, standard_registry};
use onyxia_onnx::{
    AttributeValue, DataType, Dimension, Graph, Node, TensorInfo, TensorKind, TensorShape,
};
use std::collections::HashMap;

// ── graph-building helpers ──────────────────────────────────────────────

fn input(g: &mut Graph, name: &str, dtype: DataType, shape: TensorShape) {
    g.add_tensor(TensorInfo {
        name: name.into(),
        dtype,
        shape,
        kind: TensorKind::Input,
        initializer: None,
    });
    g.inputs.push(name.into());
}

fn output(g: &mut Graph, name: &str) {
    if g.tensor_id(name).is_err() {
        intermediate(g, name);
    }
    g.outputs.push(name.into());
}

fn intermediate(g: &mut Graph, name: &str) {
    g.add_tensor(TensorInfo {
        name: name.into(),
        dtype: DataType::F32,
        shape: TensorShape::Unknown,
        kind: TensorKind::Intermediate,
        initializer: None,
    });
}

fn weight_i64(g: &mut Graph, name: &str, values: &[i64], dims: &[usize]) {
    g.add_tensor(TensorInfo {
        name: name.into(),
        dtype: DataType::I64,
        shape: TensorShape::Static(dims.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    });
}

fn weight_f32(g: &mut Graph, name: &str, values: &[f32], dims: &[usize]) {
    g.add_tensor(TensorInfo {
        name: name.into(),
        dtype: DataType::F32,
        shape: TensorShape::Static(dims.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(values.iter().flat_map(|v| v.to_le_bytes()).collect()),
    });
}

fn weight_u8(g: &mut Graph, name: &str, bytes: Vec<u8>, dims: &[usize]) {
    g.add_tensor(TensorInfo {
        name: name.into(),
        dtype: DataType::U8,
        shape: TensorShape::Static(dims.to_vec()),
        kind: TensorKind::Weight,
        initializer: Some(bytes),
    });
}

fn node(g: &mut Graph, op: &str, inputs: &[&str], outputs: &[&str]) {
    node_with(g, op, inputs, outputs, HashMap::new());
}

fn node_with(
    g: &mut Graph,
    op: &str,
    inputs: &[&str],
    outputs: &[&str],
    attributes: HashMap<String, AttributeValue>,
) {
    for out in outputs {
        if g.tensor_id(out).is_err() {
            intermediate(g, out);
        }
    }
    let mut n = Node::new(op)
        .with_inputs(inputs.iter().map(|s| s.to_string()).collect())
        .with_outputs(outputs.iter().map(|s| s.to_string()).collect());
    n.attributes = attributes;
    n.name = format!("/test/{op}");
    if op == "GroupQueryAttention" || op == "MatMulNBits" || op.contains("Rotary") {
        n.domain = "com.microsoft".into();
    }
    g.add_node(n);
}

fn dyn_shape(dims: &[&str]) -> TensorShape {
    TensorShape::Dynamic(
        dims.iter()
            .map(|d| match d.parse::<usize>() {
                Ok(v) => Dimension::Static(v),
                Err(_) => Dimension::Named(d.to_string()),
            })
            .collect(),
    )
}

// ── tests ───────────────────────────────────────────────────────────────

#[test]
fn lowers_and_evaluates_simple_graph() {
    // out = (x + w) * x
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, TensorShape::Static(vec![2, 2]));
    weight_f32(&mut g, "w", &[10., 20., 30., 40.], &[2, 2]);
    node(&mut g, "Add", &["x", "w"], &["sum"]);
    node(&mut g, "Mul", &["sum", "x"], &["out"]);
    output(&mut g, "out");

    let module = lower(g, &standard_registry()).unwrap();
    assert_eq!(module.nodes.len(), 2);

    let x = Tensor::from_f32(&[1., 2., 3., 4.], &[2, 2]).unwrap();
    let outs = eval(&module, &[("x", x)]).unwrap();
    assert_eq!(outs[0].1.to_f32().unwrap(), vec![11., 44., 99., 176.]);
}

/// B3 acceptance: the Gemma-style token-count chain
/// `Shape → Gather → Unsqueeze → Concat → Reshape` lowers to a single
/// Reshape with a symbolic target — zero runtime shape nodes.
#[test]
fn gemma_reshape_chain_folds_to_one_node() {
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, dyn_shape(&["1", "S", "256"]));
    weight_i64(&mut g, "idx1", &[1], &[]);
    weight_i64(&mut g, "axes0", &[0], &[1]);
    weight_i64(&mut g, "c1", &[1], &[1]);
    weight_i64(&mut g, "c4", &[4], &[1]);
    weight_i64(&mut g, "c64", &[64], &[1]);

    node(&mut g, "Shape", &["x"], &["shp"]);
    node(&mut g, "Gather", &["shp", "idx1"], &["s_scalar"]);
    node(&mut g, "Unsqueeze", &["s_scalar", "axes0"], &["s_vec"]);
    node_with(
        &mut g,
        "Concat",
        &["c1", "s_vec", "c4", "c64"],
        &["target"],
        HashMap::from([("axis".into(), AttributeValue::Int(0))]),
    );
    node(&mut g, "Reshape", &["x", "target"], &["out"]);
    output(&mut g, "out");

    let (module, stats) = lower_with_stats(g, &standard_registry()).unwrap();
    assert_eq!(
        stats.ir_nodes,
        1,
        "shape chain must fold away; got:\n{}",
        onyxia_ir::dot::to_dot(&module)
    );
    let NodeKind::Prim(Prim::Reshape { shape }) = &module.nodes[0].kind else {
        panic!("expected a Reshape, got {:?}", module.nodes[0].kind);
    };
    // Target is [1, S, 4, 64] with S symbolic.
    assert_eq!(shape.len(), 4);
    assert_eq!(shape[0].as_const(), Some(1));
    assert!(shape[1].as_const().is_none(), "S stays symbolic");
    assert_eq!(shape[2].as_const(), Some(4));
    assert_eq!(shape[3].as_const(), Some(64));

    // And it runs: S binds to 2 at eval time.
    let x = Tensor::from_f32(&vec![0.5; 512], &[1, 2, 256]).unwrap();
    let outs = eval(&module, &[("x", x)]).unwrap();
    assert_eq!(outs[0].1.shape(), &[1, 2, 4, 64]);
}

#[test]
fn reduce_axes_as_attr_and_as_input() {
    // Old style: axes attribute.
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, TensorShape::Static(vec![2, 3]));
    node_with(
        &mut g,
        "ReduceMean",
        &["x"],
        &["out"],
        HashMap::from([
            ("axes".into(), AttributeValue::Ints(vec![-1])),
            ("keepdims".into(), AttributeValue::Int(0)),
        ]),
    );
    output(&mut g, "out");
    let module = lower(g, &standard_registry()).unwrap();
    let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
    let outs = eval(&module, &[("x", x)]).unwrap();
    assert_eq!(outs[0].1.to_f32().unwrap(), vec![2., 5.]);

    // New style: axes as a second input.
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, TensorShape::Static(vec![2, 3]));
    weight_i64(&mut g, "axes", &[0], &[1]);
    node_with(
        &mut g,
        "ReduceSum",
        &["x", "axes"],
        &["out"],
        HashMap::from([("keepdims".into(), AttributeValue::Int(1))]),
    );
    output(&mut g, "out");
    let module = lower(g, &standard_registry()).unwrap();
    let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6.], &[2, 3]).unwrap();
    let outs = eval(&module, &[("x", x)]).unwrap();
    assert_eq!(outs[0].1.shape(), &[1, 3]);
    assert_eq!(outs[0].1.to_f32().unwrap(), vec![5., 7., 9.]);
}

#[test]
fn softmax_lowers_to_composite_and_inlines() {
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, TensorShape::Static(vec![1, 3]));
    node_with(
        &mut g,
        "Softmax",
        &["x"],
        &["out"],
        HashMap::from([("axis".into(), AttributeValue::Int(-1))]),
    );
    output(&mut g, "out");

    let module = lower(g, &standard_registry()).unwrap();
    assert!(matches!(&module.nodes[0].kind, NodeKind::Composite(c) if c.name == "Softmax"));

    let module = inline_composites(module, &standard_decompositions(), &|_| false).unwrap();
    let x = Tensor::from_f32(&[0.0, 1.0, 2.0], &[1, 3]).unwrap();
    let outs = eval(&module, &[("x", x)]).unwrap();
    let got = outs[0].1.to_f32().unwrap();
    let e: Vec<f32> = [0.0f32, 1.0, 2.0].iter().map(|v| v.exp()).collect();
    let s: f32 = e.iter().sum();
    for (a, b) in got.iter().zip(e.iter().map(|v| v / s)) {
        assert!((a - b).abs() < 1e-6);
    }
}

#[test]
fn matmul_nbits_end_to_end() {
    // Same numbers as the decomposition unit test, but arriving through a
    // real ONNX graph: K=8, N=2, block_size=4, weights packed as a u8 blob.
    let (k_dim, n_dim, bs) = (8usize, 2usize, 4usize);
    let q_vals: Vec<u8> = (0..n_dim * k_dim).map(|i| (i % 16) as u8).collect();
    let scales: Vec<f32> = vec![0.5, 0.25, 1.0, 2.0];
    let mut packed = vec![0u8; n_dim * k_dim / 2];
    for (i, &v) in q_vals.iter().enumerate() {
        packed[i / 2] |= (v & 0xF) << ((i % 2) * 4);
    }

    let mut g = Graph::new();
    input(
        &mut g,
        "a",
        DataType::F32,
        TensorShape::Static(vec![1, k_dim]),
    );
    // Blob layout as ONNX ships it: [N, n_blocks, blob_bytes] u8.
    weight_u8(&mut g, "b_q", packed, &[n_dim, k_dim / bs, bs / 2]);
    weight_f32(&mut g, "scales", &scales, &[n_dim * (k_dim / bs)]);
    node_with(
        &mut g,
        "MatMulNBits",
        &["a", "b_q", "scales"],
        &["out"],
        HashMap::from([
            ("K".into(), AttributeValue::Int(k_dim as i64)),
            ("N".into(), AttributeValue::Int(n_dim as i64)),
            ("bits".into(), AttributeValue::Int(4)),
            ("block_size".into(), AttributeValue::Int(bs as i64)),
        ]),
    );
    output(&mut g, "out");

    let module = lower(g, &standard_registry()).unwrap();
    let module = inline_composites(module, &standard_decompositions(), &|_| false).unwrap();

    let a_vals: Vec<f32> = (0..k_dim).map(|i| (i as f32) * 0.1).collect();
    let outs = eval(
        &module,
        &[("a", Tensor::from_f32(&a_vals, &[1, k_dim]).unwrap())],
    )
    .unwrap();

    let n_blocks = k_dim / bs;
    let mut expect = vec![0.0f32; n_dim];
    for n in 0..n_dim {
        for kk in 0..k_dim {
            let block = n * n_blocks + kk / bs;
            let w = (q_vals[n * k_dim + kk] as f32 - 8.0) * scales[block];
            expect[n] += a_vals[kk] * w;
        }
    }
    let got = outs[0].1.to_f32().unwrap();
    for (a, e) in got.iter().zip(&expect) {
        assert!((a - e).abs() < 1e-4, "{got:?} vs {expect:?}");
    }
}

#[test]
fn constant_of_shape_and_expand() {
    // ConstantOfShape over a symbolic shape, then Expand-broadcast a bias.
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, dyn_shape(&["S", "4"]));
    weight_f32(&mut g, "bias", &[1., 2., 3., 4.], &[4]);
    node(&mut g, "Shape", &["x"], &["shp"]);
    node(&mut g, "ConstantOfShape", &["shp"], &["zeros"]);
    node(&mut g, "Expand", &["bias", "shp"], &["bias_full"]);
    node(&mut g, "Add", &["zeros", "bias_full"], &["out"]);
    output(&mut g, "out");

    let module = lower(g, &standard_registry()).unwrap();
    let x = Tensor::from_f32(&[0.; 8], &[2, 4]).unwrap();
    let outs = eval(&module, &[("x", x)]).unwrap();
    assert_eq!(outs[0].1.shape(), &[2, 4]);
    assert_eq!(
        outs[0].1.to_f32().unwrap(),
        vec![1., 2., 3., 4., 1., 2., 3., 4.]
    );
}

#[test]
fn slice_with_sentinel_and_negative_bounds() {
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, TensorShape::Static(vec![5]));
    weight_i64(&mut g, "starts", &[1], &[1]);
    weight_i64(&mut g, "ends", &[i64::MAX], &[1]); // "to the end" sentinel
    node(&mut g, "Slice", &["x", "starts", "ends"], &["out"]);
    output(&mut g, "out");
    let module = lower(g, &standard_registry()).unwrap();
    let x = Tensor::from_f32(&[10., 11., 12., 13., 14.], &[5]).unwrap();
    let outs = eval(&module, &[("x", x)]).unwrap();
    assert_eq!(outs[0].1.to_f32().unwrap(), vec![11., 12., 13., 14.]);
}

#[test]
fn unknown_op_reports_cleanly() {
    let mut g = Graph::new();
    input(&mut g, "x", DataType::F32, TensorShape::Static(vec![1]));
    node(&mut g, "TotallyMadeUpOp", &["x"], &["out"]);
    output(&mut g, "out");
    let err = lower(g, &standard_registry()).unwrap_err().to_string();
    assert!(err.contains("TotallyMadeUpOp"), "got: {err}");
    assert!(err.contains("no lowering rule"), "got: {err}");
}

#[test]
fn gqa_through_lowering_runs() {
    // Symbolic S and T all the way from ONNX shapes through GQA inlining.
    let (h, kv, d) = (2i64, 1i64, 2usize);
    let hidden = (h as usize) * d;
    let kv_hidden = (kv as usize) * d;
    let mut g = Graph::new();
    input(
        &mut g,
        "q",
        DataType::F32,
        dyn_shape(&["1", "S", &hidden.to_string()]),
    );
    input(
        &mut g,
        "k",
        DataType::F32,
        dyn_shape(&["1", "S", &kv_hidden.to_string()]),
    );
    input(
        &mut g,
        "v",
        DataType::F32,
        dyn_shape(&["1", "S", &kv_hidden.to_string()]),
    );
    input(
        &mut g,
        "past_k",
        DataType::F32,
        dyn_shape(&["1", &kv.to_string(), "T", &d.to_string()]),
    );
    input(
        &mut g,
        "past_v",
        DataType::F32,
        dyn_shape(&["1", &kv.to_string(), "T", &d.to_string()]),
    );
    input(
        &mut g,
        "seqlens_k",
        DataType::I32,
        TensorShape::Static(vec![1]),
    );
    input(
        &mut g,
        "total_len",
        DataType::I32,
        TensorShape::Static(vec![1]),
    );
    node_with(
        &mut g,
        "GroupQueryAttention",
        &["q", "k", "v", "past_k", "past_v", "seqlens_k", "total_len"],
        &["out", "present_k", "present_v"],
        HashMap::from([
            ("num_heads".into(), AttributeValue::Int(h)),
            ("kv_num_heads".into(), AttributeValue::Int(kv)),
        ]),
    );
    output(&mut g, "out");
    output(&mut g, "present_k");
    output(&mut g, "present_v");

    let module = lower(g, &standard_registry()).unwrap();
    let module = inline_composites(module, &standard_decompositions(), &|_| false).unwrap();

    // T=0 (empty past), S=1: attention over a single position returns v.
    let mk = |vals: &[f32], shape: &[usize]| Tensor::from_f32(vals, shape).unwrap();
    let outs = eval(
        &module,
        &[
            ("q", mk(&[0.3, -0.1, 0.5, 0.2], &[1, 1, 4])),
            ("k", mk(&[0.7, 0.7], &[1, 1, 2])),
            ("v", mk(&[42.0, 7.0], &[1, 1, 2])),
            ("past_k", mk(&[], &[1, 1, 0, 2])),
            ("past_v", mk(&[], &[1, 1, 0, 2])),
            (
                "seqlens_k",
                Tensor::new(
                    onyxia_ir::DataType::I32,
                    vec![1],
                    0i32.to_le_bytes().to_vec(),
                )
                .unwrap(),
            ),
            (
                "total_len",
                Tensor::new(
                    onyxia_ir::DataType::I32,
                    vec![1],
                    1i32.to_le_bytes().to_vec(),
                )
                .unwrap(),
            ),
        ],
    )
    .unwrap();
    // Both query heads attend to the single kv position: out = [v, v].
    assert_eq!(outs[0].1.to_f32().unwrap(), vec![42.0, 7.0, 42.0, 7.0]);
    assert_eq!(outs[1].1.to_f32().unwrap(), vec![0.7, 0.7]);
    assert_eq!(outs[2].1.to_f32().unwrap(), vec![42.0, 7.0]);
}
