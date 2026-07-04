//! Composite decompositions: the portable half of the two-registry design.
//!
//! Every composite must have a decomposition into primitives (and/or other
//! composites) registered here. A backend executes a composite either with
//! a hand-written kernel from its own registry, or by asking
//! [`inline_composites`] to expand it — recursively — until only nodes it
//! supports remain. A brand-new backend with zero
//! kernels runs everything through these decompositions; they are also the
//! correctness reference that fused kernels differential-test against.
//!
//! Decompositions are pure IR: they know nothing about ONNX or any backend.

use crate::builder::GraphBuilder;
use crate::dim::DimExpr;
use crate::graph::{Composite, Module, NodeId, NodeKind, ValueId};
use crate::prim::{BinaryOp, CmpOp, Prim, ReduceOp, SliceSpec, UnaryOp};
use crate::types::DataType;
use crate::{Error, Result};
use std::collections::HashMap;

/// A decomposition: given the composite (for attrs), its input values, and
/// a builder over the containing module, emit the replacement subgraph and
/// return the values standing in for the composite's outputs.
pub type DecompFn = fn(&Composite, &[ValueId], &mut GraphBuilder) -> Result<Vec<ValueId>>;

/// Registry of decompositions, keyed by domain-qualified composite name.
#[derive(Default)]
pub struct DecompositionRegistry {
    map: HashMap<String, DecompFn>,
}

impl DecompositionRegistry {
    /// Create an empty registry.
    pub fn new() -> Self {
        Self::default()
    }

    /// Register a decomposition (replacing any previous one of that name).
    pub fn register(&mut self, name: &str, f: DecompFn) {
        self.map.insert(name.to_string(), f);
    }

    /// Look up by name.
    pub fn get(&self, name: &str) -> Option<DecompFn> {
        self.map.get(name).copied()
    }
}

/// The decompositions shipped with onyxia, covering the composite set the
/// standard ONNX lowering rules emit.
pub fn standard_decompositions() -> DecompositionRegistry {
    let mut r = DecompositionRegistry::new();
    r.register("Softmax", softmax);
    r.register("Gelu", gelu);
    r.register("Trilu", trilu);
    r.register("SimplifiedLayerNormalization", simplified_layer_norm);
    r.register("com.microsoft.RotaryEmbedding", rotary_embedding);
    r.register("com.microsoft.GemmaRotaryEmbedding", rotary_embedding);
    r.register("com.microsoft.GroupQueryAttention", group_query_attention);
    r.register("com.microsoft.MatMulNBits", matmul_nbits);
    r
}

/// Inline composites until every remaining node satisfies `supports`
/// (primitives always do). This is legalization: a backend passes its
/// kernel-registry membership as `supports`; passing `|_| false` yields a
/// pure-primitive module (what the reference backend and the interpreter
/// need).
///
/// Each inlined composite's declared output types are checked against the
/// types inferred through its decomposition — a mismatch is a bug in one of
/// the two and errors immediately.
pub fn inline_composites(
    mut module: Module,
    registry: &DecompositionRegistry,
    supports: &dyn Fn(&str) -> bool,
) -> Result<Module> {
    loop {
        // Find the first composite that must be expanded.
        let target: Option<NodeId> = module.node_ids().find(
            |&id| matches!(&module.node(id).kind, NodeKind::Composite(c) if !supports(&c.name)),
        );
        let Some(node_id) = target else {
            crate::fold::eliminate_dead(&mut module);
            return Ok(module);
        };

        let node = module.node(node_id);
        let NodeKind::Composite(composite) = node.kind.clone() else {
            unreachable!("target filtered to composites");
        };
        let inputs = node.inputs.clone();
        let old_outputs = node.outputs.clone();
        let loc = node.loc.clone();

        let decomp = registry.get(&composite.name).ok_or_else(|| {
            Error::Unsupported(format!(
                "no kernel and no decomposition for composite '{}'{}",
                composite.name,
                loc.name
                    .as_deref()
                    .map(|n| format!(" (node {n})"))
                    .unwrap_or_default()
            ))
        })?;

        let mut builder = GraphBuilder::from_module(module);
        builder.set_loc(loc);
        let new_outputs = decomp(&composite, &inputs, &mut builder)?;
        module = builder.into_module();

        if new_outputs.len() != old_outputs.len() {
            return Err(Error::InvalidGraph(format!(
                "decomposition of '{}' produced {} outputs, composite declares {}",
                composite.name,
                new_outputs.len(),
                old_outputs.len()
            )));
        }
        for (&old, &new) in old_outputs.iter().zip(&new_outputs) {
            let (old_ty, new_ty) = (&module.value(old).ty, &module.value(new).ty);
            if old_ty != new_ty {
                return Err(Error::InvalidGraph(format!(
                    "decomposition of '{}': declared output type {old_ty} \
                     but decomposition produces {new_ty}",
                    composite.name
                )));
            }
            module.replace_uses(old, new);
        }
        module.remove_nodes(&std::collections::HashSet::from([node_id]));
    }
}

// ───────────────────────── shared helpers ──────────────────────────────

/// Scalar constant of the given dtype (f64 value rounded appropriately).
fn scalar(b: &mut GraphBuilder, dtype: DataType, v: f64) -> Result<ValueId> {
    let bytes: Vec<u8> = match dtype {
        DataType::F32 => (v as f32).to_le_bytes().to_vec(),
        DataType::F16 => half::f16::from_f64(v).to_le_bytes().to_vec(),
        DataType::I64 => (v as i64).to_le_bytes().to_vec(),
        DataType::I32 => (v as i32).to_le_bytes().to_vec(),
        _ => {
            return Err(Error::Unsupported(format!(
                "scalar constant of dtype {dtype}"
            )));
        }
    };
    b.constant(crate::types::TensorType::of(dtype, &[]), bytes)
}

/// Slice one axis with step 1.
fn slice1(
    b: &mut GraphBuilder,
    x: ValueId,
    axis: usize,
    start: DimExpr,
    end: DimExpr,
) -> Result<ValueId> {
    b.slice(
        x,
        vec![SliceSpec {
            axis,
            start,
            end,
            step: 1,
        }],
    )
}

// ─────────────────────────── decompositions ────────────────────────────

/// Softmax(axis): numerically stable max-subtracted form.
fn softmax(c: &Composite, inputs: &[ValueId], b: &mut GraphBuilder) -> Result<Vec<ValueId>> {
    let x = inputs[0];
    let axis = c.attrs.int("axis")? as usize; // normalized by lowering
    let mx = b.reduce(ReduceOp::Max, x, &[axis], true)?;
    let shifted = b.sub(x, mx)?;
    let e = b.unary(UnaryOp::Exp, shifted)?;
    let sum = b.reduce(ReduceOp::Sum, e, &[axis], true)?;
    Ok(vec![b.div(e, sum)?])
}

/// Gelu: exact (erf) or tanh approximation, per the `approximate` attr.
fn gelu(c: &Composite, inputs: &[ValueId], b: &mut GraphBuilder) -> Result<Vec<ValueId>> {
    let x = inputs[0];
    let dt = b.ty(x).dtype;
    let half_c = scalar(b, dt, 0.5)?;
    let one = scalar(b, dt, 1.0)?;
    let inner = match c.attrs.str("approximate").unwrap_or("none") {
        "tanh" => {
            // 0.5x(1 + tanh(√(2/π)(x + 0.044715x³)))
            let coef = scalar(b, dt, 0.044715)?;
            let sqrt_2_over_pi = scalar(b, dt, (2.0 / std::f64::consts::PI).sqrt())?;
            let x2 = b.mul(x, x)?;
            let x3 = b.mul(x2, x)?;
            let cx3 = b.mul(coef, x3)?;
            let sum = b.add(x, cx3)?;
            let scaled = b.mul(sqrt_2_over_pi, sum)?;
            b.unary(UnaryOp::Tanh, scaled)?
        }
        _ => {
            // 0.5x(1 + erf(x/√2))
            let inv_sqrt2 = scalar(b, dt, std::f64::consts::FRAC_1_SQRT_2)?;
            let scaled = b.mul(x, inv_sqrt2)?;
            b.unary(UnaryOp::Erf, scaled)?
        }
    };
    let one_plus = b.add(one, inner)?;
    let hx = b.mul(half_c, x)?;
    Ok(vec![b.mul(hx, one_plus)?])
}

/// Trilu(upper, k): keep the upper/lower triangle, zero the rest.
fn trilu(c: &Composite, inputs: &[ValueId], b: &mut GraphBuilder) -> Result<Vec<ValueId>> {
    let x = inputs[0];
    let upper = c.attrs.int_or("upper", 1)? != 0;
    let k = c.attrs.int_or("k", 0)?;
    let dims = b.ty(x).shape.dims().to_vec();
    if dims.len() < 2 {
        return Err(Error::Shape("Trilu requires rank >= 2".into()));
    }
    let (r, cdim) = (dims[dims.len() - 2].clone(), dims[dims.len() - 1].clone());
    let rows = b.iota(r.clone(), DataType::I64)?;
    let rows = b.reshape(rows, vec![r, DimExpr::constant(1)])?;
    let cols = b.iota(cdim.clone(), DataType::I64)?;
    let cols = b.reshape(cols, vec![DimExpr::constant(1), cdim])?;
    let k_c = scalar(b, DataType::I64, k as f64)?;
    let shifted_rows = b.add(rows, k_c)?;
    // upper: keep col >= row + k;  lower: keep col <= row + k.
    let keep = b.cmp(
        if upper { CmpOp::Ge } else { CmpOp::Le },
        cols,
        shifted_rows,
    )?;
    let zero = scalar(b, b.ty(x).dtype, 0.0)?;
    Ok(vec![b.select(keep, x, zero)?])
}

/// SimplifiedLayerNormalization (RMS norm): x * rsqrt(mean(x², -1) + ε) * w.
fn simplified_layer_norm(
    c: &Composite,
    inputs: &[ValueId],
    b: &mut GraphBuilder,
) -> Result<Vec<ValueId>> {
    let (x, w) = (inputs[0], inputs[1]);
    let eps = c.attrs.float_or("epsilon", 1e-5)?;
    let last = b.ty(x).shape.rank() - 1;
    let sq = b.mul(x, x)?;
    let ms = b.reduce(ReduceOp::Mean, sq, &[last], true)?;
    let eps_c = scalar(b, b.ty(x).dtype, eps)?;
    let denom = b.add(ms, eps_c)?;
    let inv = b.unary(UnaryOp::Rsqrt, denom)?;
    let normed = b.mul(x, inv)?;
    Ok(vec![b.mul(normed, w)?])
}

/// com.microsoft.RotaryEmbedding, non-interleaved.
///
/// input `[B, S, H*D]`, position_ids `[B, S]` (i64), cos/sin caches
/// `[max_seq, R/2]`. Rotates the first `R` dims of each head; the rest pass
/// through. `interleaved=1` is not yet supported (Gemma does not use it).
fn rotary_embedding(
    c: &Composite,
    inputs: &[ValueId],
    b: &mut GraphBuilder,
) -> Result<Vec<ValueId>> {
    if c.attrs.int_or("interleaved", 0)? != 0 {
        return Err(Error::Unsupported(
            "RotaryEmbedding interleaved=1 decomposition".into(),
        ));
    }
    let (x, pos, cos_cache, sin_cache) = (inputs[0], inputs[1], inputs[2], inputs[3]);
    let x_dims = b.ty(x).shape.dims().to_vec();
    if x_dims.len() != 3 {
        return Err(Error::Unsupported(format!(
            "RotaryEmbedding expects 3-D input, got {}",
            b.ty(x).shape
        )));
    }
    let (bsz, seq, hidden) = (x_dims[0].clone(), x_dims[1].clone(), x_dims[2].clone());
    // num_heads=0 (the common export) means "infer": the cache fixes the
    // rotary width R; when hidden splits evenly into R-wide heads, rotation
    // is per head (matches the old kernel and onnxruntime).
    let heads = match c.attrs.int_or("num_heads", 0)? {
        n if n > 0 => n as u64,
        _ => {
            let cache_r = b.ty(cos_cache).shape.dims()[1]
                .as_const()
                .map(|half| half * 2);
            match (hidden.as_const(), cache_r) {
                (Some(h), Some(r)) if r > 0 && h % r == 0 => h / r,
                _ => 1,
            }
        }
    };
    let d = hidden
        .clone()
        .div_exact(&DimExpr::constant(heads))
        .ok_or_else(|| {
            Error::Shape(format!(
                "hidden dim {hidden} not divisible by num_heads {heads}"
            ))
        })?;
    // R/2 comes from the cache; R from the attr when present.
    let half_r = match c.attrs.int_or("rotary_embedding_dim", 0)? {
        0 => b.ty(cos_cache).shape.dims()[1].clone(),
        r => DimExpr::constant((r as u64) / 2),
    };
    let r_full = half_r.clone() * DimExpr::constant(2);

    // Gathered caches: [B, S, R/2] → [B, S, 1, R/2] to broadcast over heads.
    let bc_shape = vec![
        bsz.clone(),
        seq.clone(),
        DimExpr::constant(1),
        half_r.clone(),
    ];
    let cos = b.gather(cos_cache, pos, 0)?;
    let cos = b.reshape(cos, bc_shape.clone())?;
    let sin = b.gather(sin_cache, pos, 0)?;
    let sin = b.reshape(sin, bc_shape)?;

    // Split heads and rotary halves.
    let x4 = b.reshape(
        x,
        vec![
            bsz.clone(),
            seq.clone(),
            DimExpr::constant(heads),
            d.clone(),
        ],
    )?;
    let x1 = slice1(b, x4, 3, DimExpr::constant(0), half_r.clone())?;
    let x2 = slice1(b, x4, 3, half_r.clone(), r_full.clone())?;

    // out1 = x1·cos − x2·sin;  out2 = x2·cos + x1·sin.
    let x1c = b.mul(x1, cos)?;
    let x2s = b.mul(x2, sin)?;
    let o1 = b.sub(x1c, x2s)?;
    let x2c = b.mul(x2, cos)?;
    let x1s = b.mul(x1, sin)?;
    let o2 = b.add(x2c, x1s)?;

    let rest_len = d.clone() - r_full.clone();
    let parts: Vec<ValueId> = if rest_len.as_const() == Some(0) {
        vec![o1, o2]
    } else {
        let rest = slice1(b, x4, 3, r_full, d)?;
        vec![o1, o2, rest]
    };
    let joined = b.concat(&parts, 3)?;
    Ok(vec![b.reshape(joined, vec![bsz, seq, hidden])?])
}

/// com.microsoft.GroupQueryAttention.
///
/// query `[B, S, H*D]`, key/value `[B, S, KV*D]`, past_key/past_value
/// `[B, KV, P, D]` (BNSH), seqlens_k `int32` with `B` elements. Outputs:
/// attention `[B, S, H*D]` and present key/value `[B, KV, P+S, D]`.
///
/// `seqlens_k` is honored per batch row, matching the onnxruntime CPU
/// reference (`gqa_attention_base.h`): row `b`'s valid length is
/// `seqlens_k[b] + 1`, its past length is `max(that - S, 0)`, new keys
/// land right after the valid past, and everything beyond the valid
/// length is zeroed in the present cache and masked in the attention.
/// Causal masking with optional sliding window (`local_window_size`,
/// which counts *previous* tokens: a query sees `window + 1` keys
/// including itself, the onnxruntime convention).
fn group_query_attention(
    c: &Composite,
    inputs: &[ValueId],
    b: &mut GraphBuilder,
) -> Result<Vec<ValueId>> {
    let (q, k, v, past_k, past_v, seqlens) = (
        inputs[0], inputs[1], inputs[2], inputs[3], inputs[4], inputs[5],
    );
    let heads = c.attrs.int("num_heads")? as u64;
    let kv_heads = c.attrs.int("kv_num_heads")? as u64;
    if heads == 0 || kv_heads == 0 || heads % kv_heads != 0 {
        return Err(Error::Attribute(format!(
            "GQA num_heads={heads} must be a positive multiple of kv_num_heads={kv_heads}"
        )));
    }
    let group = heads / kv_heads;
    let window = c.attrs.int_or("local_window_size", -1)?;

    let q_dims = b.ty(q).shape.dims().to_vec();
    let (bsz, seq, hidden) = (q_dims[0].clone(), q_dims[1].clone(), q_dims[2].clone());
    let d = hidden
        .clone()
        .div_exact(&DimExpr::constant(heads))
        .ok_or_else(|| {
            Error::Shape(format!(
                "GQA hidden {hidden} not divisible by num_heads {heads}"
            ))
        })?;
    let past = b.ty(past_k).shape.dims()[2].clone();
    let total = past.clone() + seq.clone();
    let dt = b.ty(q).dtype;

    // Per-row valid lengths: tot_b = seqlens_k[b] + 1 (shape [B,1,1,1]
    // i64), past_b = max(tot_b - S, 0).
    let one4 = DimExpr::constant(1);
    let sl = b.reshape(
        seqlens,
        vec![bsz.clone(), one4.clone(), one4.clone(), one4.clone()],
    )?;
    let sl = b.cast(sl, DataType::I64)?;
    let one_i = scalar(b, DataType::I64, 1.0)?;
    let tot_b = b.add(sl, one_i)?;
    let s_dim = b.prim(
        Prim::DimValues {
            exprs: vec![seq.clone()],
        },
        &[],
    )?;
    let past_b_raw = b.sub(tot_b, s_dim)?;
    let zero_i = scalar(b, DataType::I64, 0.0)?;
    let past_b = b.prim(Prim::Binary(BinaryOp::Max), &[past_b_raw, zero_i])?;
    let jrel = b.iota(seq.clone(), DataType::I64)?;
    let jrel = b.reshape(
        jrel,
        vec![one4.clone(), one4.clone(), seq.clone(), one4.clone()],
    )?;

    // Fused rotary: positions come from the per-row past length, exactly
    // like the masking (position_ids input is rejected at lowering).
    let (mut q, mut k) = (q, k);
    if c.attrs.int_or("do_rotary", 0)? != 0 {
        let (cos, sin) = (inputs[6], inputs[7]);
        let interleaved = c.attrs.int_or("rotary_interleaved", 0)?;
        let qpos = b.add(past_b, jrel)?; // [B,1,S,1]
        let pos2 = b.reshape(qpos, vec![bsz.clone(), seq.clone()])?;
        let rope = |b: &mut GraphBuilder, x: ValueId, n: u64| -> Result<ValueId> {
            let ty = b.ty(x).clone();
            Ok(b.composite(
                "com.microsoft.RotaryEmbedding",
                crate::attrs::Attrs::new()
                    .with("num_heads", crate::attrs::AttrValue::Int(n as i64))
                    .with("interleaved", crate::attrs::AttrValue::Int(interleaved)),
                &[x, pos2, cos, sin],
                vec![ty],
            )?
            .remove(0))
        };
        q = rope(b, q, heads)?;
        k = rope(b, k, kv_heads)?;
    }

    // To BNSH: [B, S, N*D] → [B, S, N, D] → [B, N, S, D].
    let to_heads = |b: &mut GraphBuilder, x: ValueId, n: u64| -> Result<ValueId> {
        let r = b.reshape(
            x,
            vec![bsz.clone(), seq.clone(), DimExpr::constant(n), d.clone()],
        )?;
        b.transpose(r, &[0, 2, 1, 3])
    };
    let q4 = to_heads(b, q, heads)?;
    let k4 = to_heads(b, k, kv_heads)?;
    let v4 = to_heads(b, v, kv_heads)?;

    // Present caches: row b keeps its valid past `[0, past_b)`, new
    // keys/values land at `[past_b, past_b + S)`, and the tail is zero.
    // Per-row placement via a one-hot matmul (`onehot[b,t,j] = (t ==
    // past_b + j)`), which needs no scatter support from backends.
    let zero_f = scalar(b, dt, 0.0)?;
    let one_f = scalar(b, dt, 1.0)?;
    let pcols = b.iota(past.clone(), DataType::I64)?;
    let pcols = b.reshape(
        pcols,
        vec![one4.clone(), one4.clone(), past.clone(), one4.clone()],
    )?;
    let past_valid = b.cmp(CmpOp::Lt, pcols, past_b)?; // [B,1,P,1]
    let kv_c = DimExpr::constant(kv_heads);
    let new_zeros = b.broadcast(
        zero_f,
        vec![bsz.clone(), kv_c.clone(), seq.clone(), d.clone()],
    )?;
    let tcols = b.iota(total.clone(), DataType::I64)?;
    let tcols = b.reshape(
        tcols,
        vec![one4.clone(), one4.clone(), total.clone(), one4.clone()],
    )?;
    let jrow = b.reshape(
        jrel,
        vec![one4.clone(), one4.clone(), one4.clone(), seq.clone()],
    )?;
    let tpos = b.add(past_b, jrow)?; // [B,1,1,S]
    let hot = b.cmp(CmpOp::Eq, tcols, tpos)?; // [B,1,T,S]
    let onehot = b.select(hot, one_f, zero_f)?;
    // Materialize the KV batch dim: backends need not broadcast matmul
    // batch dims.
    let onehot = b.broadcast(
        onehot,
        vec![bsz.clone(), kv_c.clone(), total.clone(), seq.clone()],
    )?;

    let place_present =
        |b: &mut GraphBuilder, past_kv: ValueId, new4: ValueId| -> Result<ValueId> {
            let kept = b.select(past_valid, past_kv, zero_f)?;
            let padded = b.concat(&[kept, new_zeros], 2)?;
            let placed = b.matmul(onehot, new4)?; // [B,KV,T,D]
            b.add(padded, placed)
        };
    let present_k = place_present(b, past_k, k4)?;
    let present_v = place_present(b, past_v, v4)?;

    // Repeat KV heads up to H: [B, KV, T+S, D] → [B, KV, G, T+S, D] → [B, H, ...].
    let repeat = |b: &mut GraphBuilder, x: ValueId| -> Result<ValueId> {
        if group == 1 {
            return Ok(x);
        }
        let with_g = b.reshape(
            x,
            vec![
                bsz.clone(),
                DimExpr::constant(kv_heads),
                DimExpr::constant(1),
                total.clone(),
                d.clone(),
            ],
        )?;
        let grown = b.broadcast(
            with_g,
            vec![
                bsz.clone(),
                DimExpr::constant(kv_heads),
                DimExpr::constant(group),
                total.clone(),
                d.clone(),
            ],
        )?;
        b.reshape(
            grown,
            vec![
                bsz.clone(),
                DimExpr::constant(heads),
                total.clone(),
                d.clone(),
            ],
        )
    };
    let k_rep = repeat(b, present_k)?;
    let v_rep = repeat(b, present_v)?;

    // Scores: q·kᵀ, scaled.
    let scores = b.prim(
        Prim::MatMul {
            trans_a: false,
            trans_b: true,
        },
        &[q4, k_rep],
    )?;
    let scale_attr = c.attrs.float_or("scale", 0.0)?;
    let scale = if scale_attr == 0.0 {
        let d_const = d.as_const().ok_or_else(|| {
            Error::Shape("GQA head dim must be constant for the default scale".into())
        })?;
        1.0 / (d_const as f64).sqrt()
    } else {
        scale_attr
    };
    let scale_c = scalar(b, dt, scale)?;
    let mut scores = b.mul(scores, scale_c)?;

    // Additive attention bias `[B|1, H|1, S, T]`, broadcast over scores.
    if c.attrs.int_or("has_attention_bias", 0)? != 0 {
        let bias = *inputs.last().expect("bias input recorded by lowering");
        scores = b.add(scores, bias)?;
    }

    // Mask: query i of row b sits at absolute position past_b + i;
    // allow col ≤ qpos, and within the sliding window when configured
    // (col ≥ qpos − window: the window counts previous tokens, so
    // window + 1 keys are visible including the query itself).
    let qpos = b.add(past_b, jrel)?; // [B,1,S,1]
    let cols = b.iota(total.clone(), DataType::I64)?;
    let cols = b.reshape(
        cols,
        vec![one4.clone(), one4.clone(), one4.clone(), total.clone()],
    )?;
    let causal = b.cmp(CmpOp::Le, cols, qpos)?; // [B,1,S,T]
    let allowed = if window >= 0 {
        let w = scalar(b, DataType::I64, window as f64)?;
        let min_col = b.sub(qpos, w)?;
        let in_window = b.cmp(CmpOp::Ge, cols, min_col)?;
        b.prim(Prim::Binary(BinaryOp::And), &[causal, in_window])?
    } else {
        causal
    };
    let neg_inf = scalar(b, dt, f64::NEG_INFINITY)?;
    let masked = b.select(allowed, scores, neg_inf)?;

    // Softmax over keys — emitted as a composite so backends with a fused
    // softmax kernel keep it; otherwise it inlines recursively.
    let probs = b
        .composite(
            "Softmax",
            crate::attrs::Attrs::new().with("axis", crate::attrs::AttrValue::Int(3)),
            &[masked],
            vec![b.ty(masked).clone()],
        )?
        .remove(0);

    // Context: probs·v → back to [B, S, H*D].
    let ctx = b.matmul(probs, v_rep)?;
    let ctx = b.transpose(ctx, &[0, 2, 1, 3])?;
    let out = b.reshape(ctx, vec![bsz, seq, hidden])?;

    Ok(vec![out, present_k, present_v])
}

/// com.microsoft.MatMulNBits: A · dequantize(B)ᵀ.
///
/// Lowering has already rewrapped the packed weight blob as logical
/// `U4 [N, n_blocks, block_size]` and zero points (when present) as
/// `U4 [N, n_blocks]`; scales arrive as `[N * n_blocks]`.
fn matmul_nbits(c: &Composite, inputs: &[ValueId], b: &mut GraphBuilder) -> Result<Vec<ValueId>> {
    let (a, bq, scales) = (inputs[0], inputs[1], inputs[2]);
    let bits = c.attrs.int_or("bits", 4)?;
    if bits != 4 {
        return Err(Error::Unsupported(format!("MatMulNBits with bits={bits}")));
    }
    let k = c.attrs.int("K")? as u64;
    let n = c.attrs.int("N")? as u64;
    let block_size = c.attrs.int("block_size")? as usize;

    let n_blocks = b.ty(bq).shape.dims()[1].clone();
    let scales2 = b.reshape(scales, vec![DimExpr::constant(n), n_blocks.clone()])?;
    let padded_k = n_blocks * DimExpr::constant(block_size as u64);

    let mut deq_inputs = vec![bq, scales2];
    if let Some(&zp) = inputs.get(3) {
        deq_inputs.push(zp);
    }
    let deq = b.prim(
        Prim::Dequantize {
            block_size,
            out_shape: vec![DimExpr::constant(n), padded_k.clone()],
        },
        &deq_inputs,
    )?;
    // Blocks pad K up to a multiple of block_size; trim the tail.
    let mut w = if padded_k.as_const() == Some(k) {
        deq
    } else {
        slice1(b, deq, 1, DimExpr::constant(0), DimExpr::constant(k))?
    };
    // Dequantized dtype follows the scales; align to the activations.
    if b.ty(w).dtype != b.ty(a).dtype {
        let to = b.ty(a).dtype;
        w = b.cast(w, to)?;
    }
    Ok(vec![b.prim(
        Prim::MatMul {
            trans_a: false,
            trans_b: true,
        },
        &[a, w],
    )?])
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attrs::{AttrValue, Attrs};
    use crate::dim::SymbolicShape;
    use crate::interp::{Tensor, eval};
    use crate::types::TensorType;

    /// Build a module with one composite, inline it fully, run it.
    fn run_composite(
        name: &str,
        attrs: Attrs,
        inputs: Vec<(&str, Tensor)>,
        out_ty: Vec<TensorType>,
    ) -> Vec<(String, Tensor)> {
        let mut b = GraphBuilder::new();
        let ids: Vec<ValueId> = inputs
            .iter()
            .map(|(n, t)| {
                b.input(
                    n,
                    TensorType::of(
                        t.dtype(),
                        &t.shape().iter().map(|&d| d as u64).collect::<Vec<_>>(),
                    ),
                )
            })
            .collect();
        let outs = b.composite(name, attrs, &ids, out_ty).unwrap();
        for (i, &o) in outs.iter().enumerate() {
            b.output(&format!("out{i}"), o);
        }
        let module = b.finish().unwrap();
        let module = inline_composites(module, &standard_decompositions(), &|_| false).unwrap();
        crate::validate::validate(&module).unwrap();
        // The inlined module must be pure primitives.
        assert!(
            module
                .nodes
                .iter()
                .all(|n| matches!(n.kind, NodeKind::Prim(_))),
            "composites remain after inlining"
        );
        eval(
            &module,
            &inputs
                .iter()
                .map(|(n, t)| (*n, t.clone()))
                .collect::<Vec<_>>(),
        )
        .unwrap()
    }

    fn assert_close(actual: &[f32], expect: &[f32], tol: f32) {
        assert_eq!(actual.len(), expect.len());
        for (i, (a, e)) in actual.iter().zip(expect).enumerate() {
            assert!(
                (a - e).abs() <= tol + e.abs() * tol,
                "element {i}: {a} vs {e}"
            );
        }
    }

    #[test]
    fn softmax_matches_manual() {
        let x = Tensor::from_f32(&[1.0, 2.0, 3.0, 1.0, 1.0, 1.0], &[2, 3]).unwrap();
        let outs = run_composite(
            "Softmax",
            Attrs::new().with("axis", AttrValue::Int(1)),
            vec![("x", x)],
            vec![TensorType::of(DataType::F32, &[2, 3])],
        );
        let got = outs[0].1.to_f32().unwrap();
        let e: Vec<f32> = [1.0f32, 2.0, 3.0].iter().map(|v| v.exp()).collect();
        let s: f32 = e.iter().sum();
        assert_close(&got[..3], &[e[0] / s, e[1] / s, e[2] / s], 1e-6);
        assert_close(&got[3..], &[1.0 / 3.0; 3], 1e-6);
    }

    #[test]
    fn gelu_matches_reference() {
        let vals = [-2.0f32, -0.5, 0.0, 0.5, 2.0];
        let x = Tensor::from_f32(&vals, &[5]).unwrap();
        let outs = run_composite(
            "Gelu",
            Attrs::new(),
            vec![("x", x)],
            vec![TensorType::of(DataType::F32, &[5])],
        );
        // Reference values for exact GELU.
        let expect = [-0.0455, -0.1543, 0.0, 0.3457, 1.9545];
        assert_close(&outs[0].1.to_f32().unwrap(), &expect, 1e-3);
    }

    #[test]
    fn trilu_upper_and_lower() {
        let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6., 7., 8., 9.], &[3, 3]).unwrap();
        let upper = run_composite(
            "Trilu",
            Attrs::new().with("upper", AttrValue::Int(1)),
            vec![("x", x.clone())],
            vec![TensorType::of(DataType::F32, &[3, 3])],
        );
        assert_eq!(
            upper[0].1.to_f32().unwrap(),
            vec![1., 2., 3., 0., 5., 6., 0., 0., 9.]
        );
        let lower_k1 = run_composite(
            "Trilu",
            Attrs::new()
                .with("upper", AttrValue::Int(0))
                .with("k", AttrValue::Int(1)),
            vec![("x", x)],
            vec![TensorType::of(DataType::F32, &[3, 3])],
        );
        assert_eq!(
            lower_k1[0].1.to_f32().unwrap(),
            vec![1., 2., 0., 4., 5., 6., 7., 8., 9.]
        );
    }

    #[test]
    fn rms_norm_matches_manual() {
        let x = Tensor::from_f32(&[1., 2., 3., 4.], &[1, 4]).unwrap();
        let w = Tensor::from_f32(&[2., 2., 2., 2.], &[4]).unwrap();
        let outs = run_composite(
            "SimplifiedLayerNormalization",
            Attrs::new().with("epsilon", AttrValue::Float(1e-6)),
            vec![("x", x), ("w", w)],
            vec![TensorType::of(DataType::F32, &[1, 4])],
        );
        let ms = (1.0 + 4.0 + 9.0 + 16.0) / 4.0;
        let inv = 1.0 / (ms + 1e-6f32).sqrt();
        let expect: Vec<f32> = [1., 2., 3., 4.].iter().map(|v| v * inv * 2.0).collect();
        assert_close(&outs[0].1.to_f32().unwrap(), &expect, 1e-5);
    }

    #[test]
    fn rotary_rotates_first_pair() {
        // B=1, S=1, one head, D=4, R=4 (cache half = 2). Position 0 → cos=1,
        // sin=0 → identity. Position 1 with cos=0, sin=1 → (x1,x2) → (-x2, x1).
        let x = Tensor::from_f32(&[1., 2., 3., 4.], &[1, 1, 4]).unwrap();
        let cos = Tensor::from_f32(&[1., 1., 0., 0.], &[2, 2]).unwrap();
        let sin = Tensor::from_f32(&[0., 0., 1., 1.], &[2, 2]).unwrap();
        let out_ty = vec![TensorType::of(DataType::F32, &[1, 1, 4])];

        let pos0 = Tensor::from_i64(&[0], &[1, 1]).unwrap();
        let id = run_composite(
            "com.microsoft.RotaryEmbedding",
            Attrs::new().with("num_heads", AttrValue::Int(1)),
            vec![
                ("x", x.clone()),
                ("pos", pos0),
                ("cos", cos.clone()),
                ("sin", sin.clone()),
            ],
            out_ty.clone(),
        );
        assert_close(&id[0].1.to_f32().unwrap(), &[1., 2., 3., 4.], 1e-6);

        let pos1 = Tensor::from_i64(&[1], &[1, 1]).unwrap();
        let rot = run_composite(
            "com.microsoft.RotaryEmbedding",
            Attrs::new().with("num_heads", AttrValue::Int(1)),
            vec![("x", x), ("pos", pos1), ("cos", cos), ("sin", sin)],
            out_ty,
        );
        // x1=(1,2), x2=(3,4): out1 = x1·0 − x2·1 = (−3,−4); out2 = x2·0 + x1·1 = (1,2).
        assert_close(&rot[0].1.to_f32().unwrap(), &[-3., -4., 1., 2.], 1e-6);
    }

    #[test]
    fn rotary_infers_heads_from_cache_width() {
        // num_heads=0 (as exported by optimum for Gemma): hidden=8 with a
        // cache half-width of 2 (R=4) must infer 2 heads and rotate BOTH,
        // not treat the input as one 8-wide head. Regression test for a
        // whole-model parity failure where only head 0 was rotated.
        let x = Tensor::from_f32(&[1., 2., 3., 4., 5., 6., 7., 8.], &[1, 1, 8]).unwrap();
        let cos = Tensor::from_f32(&[1., 1., 0., 0.], &[2, 2]).unwrap();
        let sin = Tensor::from_f32(&[0., 0., 1., 1.], &[2, 2]).unwrap();
        let pos1 = Tensor::from_i64(&[1], &[1, 1]).unwrap();
        let rot = run_composite(
            "com.microsoft.RotaryEmbedding",
            Attrs::new().with("num_heads", AttrValue::Int(0)),
            vec![("x", x), ("pos", pos1), ("cos", cos), ("sin", sin)],
            vec![TensorType::of(DataType::F32, &[1, 1, 8])],
        );
        assert_close(
            &rot[0].1.to_f32().unwrap(),
            &[-3., -4., 1., 2., -7., -8., 5., 6.],
            1e-6,
        );
    }

    /// Naive attention in plain Rust, for GQA cross-checking.
    /// q: [S, D] (one head), kv: [T, D]; causal with `past` offset.
    fn naive_attention(
        q: &[Vec<f32>],
        keys: &[Vec<f32>],
        vals: &[Vec<f32>],
        past: usize,
        scale: f32,
        window: i64,
    ) -> Vec<Vec<f32>> {
        let d = q[0].len();
        q.iter()
            .enumerate()
            .map(|(i, qi)| {
                let row = past + i;
                let mut logits: Vec<f32> = keys
                    .iter()
                    .enumerate()
                    .map(|(j, kj)| {
                        // Window counts previous tokens (onnxruntime
                        // convention): window + 1 keys visible.
                        let visible = j <= row && (window < 0 || (j as i64) >= row as i64 - window);
                        if visible {
                            qi.iter().zip(kj).map(|(a, b)| a * b).sum::<f32>() * scale
                        } else {
                            f32::NEG_INFINITY
                        }
                    })
                    .collect();
                let m = logits.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
                logits.iter_mut().for_each(|l| *l = (*l - m).exp());
                let s: f32 = logits.iter().sum();
                (0..d)
                    .map(|c| {
                        logits
                            .iter()
                            .zip(vals)
                            .map(|(p, v)| p / s * v[c])
                            .sum::<f32>()
                    })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn gqa_matches_naive_attention() {
        // B=1, H=2, KV=1 (group=2), D=2, S=2 new tokens, T=1 past.
        let (h, kv, d, s, t) = (2usize, 1usize, 2usize, 2usize, 1usize);
        let q: Vec<f32> = (0..s * h * d).map(|i| (i as f32) * 0.1 - 0.3).collect();
        let k_new: Vec<f32> = (0..s * kv * d).map(|i| (i as f32) * 0.2 - 0.1).collect();
        let v_new: Vec<f32> = (0..s * kv * d).map(|i| (i as f32) * 0.3 + 0.2).collect();
        let past_k: Vec<f32> = (0..t * kv * d).map(|i| (i as f32) * 0.5).collect();
        let past_v: Vec<f32> = (0..t * kv * d).map(|i| 1.0 - (i as f32) * 0.4).collect();

        let outs = run_composite(
            "com.microsoft.GroupQueryAttention",
            Attrs::new()
                .with("num_heads", AttrValue::Int(h as i64))
                .with("kv_num_heads", AttrValue::Int(kv as i64)),
            vec![
                ("q", Tensor::from_f32(&q, &[1, s, h * d]).unwrap()),
                ("k", Tensor::from_f32(&k_new, &[1, s, kv * d]).unwrap()),
                ("v", Tensor::from_f32(&v_new, &[1, s, kv * d]).unwrap()),
                ("pk", Tensor::from_f32(&past_k, &[1, kv, t, d]).unwrap()),
                ("pv", Tensor::from_f32(&past_v, &[1, kv, t, d]).unwrap()),
                (
                    "seqlens",
                    Tensor::from_i64(&[(t + s - 1) as i64], &[1]).unwrap(),
                ),
                ("total", Tensor::from_i64(&[(t + s) as i64], &[1]).unwrap()),
            ],
            vec![
                TensorType::of(DataType::F32, &[1, s as u64, (h * d) as u64]),
                TensorType::of(DataType::F32, &[1, kv as u64, (t + s) as u64, d as u64]),
                TensorType::of(DataType::F32, &[1, kv as u64, (t + s) as u64, d as u64]),
            ],
        );

        // Reference: per query head, KV head 0 serves both (group=2).
        let keys: Vec<Vec<f32>> = {
            let mut all = Vec::new();
            for i in 0..t {
                all.push(past_k[i * d..(i + 1) * d].to_vec());
            }
            for i in 0..s {
                all.push(k_new[i * kv * d..i * kv * d + d].to_vec());
            }
            all
        };
        let vals: Vec<Vec<f32>> = {
            let mut all = Vec::new();
            for i in 0..t {
                all.push(past_v[i * d..(i + 1) * d].to_vec());
            }
            for i in 0..s {
                all.push(v_new[i * kv * d..i * kv * d + d].to_vec());
            }
            all
        };
        let scale = 1.0 / (d as f32).sqrt();
        let got = outs[0].1.to_f32().unwrap();
        for head in 0..h {
            let qh: Vec<Vec<f32>> = (0..s)
                .map(|i| q[i * h * d + head * d..i * h * d + head * d + d].to_vec())
                .collect();
            let expect = naive_attention(&qh, &keys, &vals, t, scale, -1);
            for i in 0..s {
                let out_row = &got[i * h * d + head * d..i * h * d + head * d + d];
                assert_close(out_row, &expect[i], 1e-4);
            }
        }
        // Present KV = past ++ new.
        let pk = outs[1].1.to_f32().unwrap();
        assert_close(&pk[..t * d], &past_k, 1e-6);
        let pv = outs[2].1.to_f32().unwrap();
        assert_close(&pv[t * d..], &v_new, 1e-6);
    }

    #[test]
    fn gqa_sliding_window_masks_old_tokens() {
        // One head, D=1, S=1 query with T=3 past, window=2. The window
        // counts previous tokens (onnxruntime convention), so window + 1
        // positions are visible: two past tokens + self.
        let outs = run_composite(
            "com.microsoft.GroupQueryAttention",
            Attrs::new()
                .with("num_heads", AttrValue::Int(1))
                .with("kv_num_heads", AttrValue::Int(1))
                .with("local_window_size", AttrValue::Int(2)),
            vec![
                ("q", Tensor::from_f32(&[1.0], &[1, 1, 1]).unwrap()),
                ("k", Tensor::from_f32(&[0.0], &[1, 1, 1]).unwrap()),
                ("v", Tensor::from_f32(&[100.0], &[1, 1, 1]).unwrap()),
                // Past keys all 0 → uniform logits among visible.
                (
                    "pk",
                    Tensor::from_f32(&[0., 0., 0.], &[1, 1, 3, 1]).unwrap(),
                ),
                (
                    "pv",
                    Tensor::from_f32(&[7., 11., 13.], &[1, 1, 3, 1]).unwrap(),
                ),
                ("seqlens", Tensor::from_i64(&[3], &[1]).unwrap()),
                ("total", Tensor::from_i64(&[4], &[1]).unwrap()),
            ],
            vec![
                TensorType::of(DataType::F32, &[1, 1, 1]),
                TensorType::of(DataType::F32, &[1, 1, 4, 1]),
                TensorType::of(DataType::F32, &[1, 1, 4, 1]),
            ],
        );
        // Visible: positions 1–2 (values 11, 13) and self (value 100),
        // uniform logits → (11 + 13 + 100) / 3.
        assert_close(&outs[0].1.to_f32().unwrap(), &[124.0 / 3.0], 1e-4);
    }

    #[test]
    fn gqa_ragged_batch_honors_seqlens() {
        // B=2 token generation (S=1) over a P=3 past buffer. Row 0 is
        // dense (seqlens_k = 3: full past). Row 1 has only 1 valid past
        // token (seqlens_k = 1); its buffer tail is garbage that must
        // not be attended, and its new key must land at position 1.
        let (kv, d, s, p) = (1usize, 2usize, 1usize, 3usize);
        let q = [[0.1f32, -0.2], [0.3, 0.05]];
        let k_new = [[0.5f32, 0.1], [-0.2, 0.4]];
        let v_new = [[10.0f32, 20.0], [30.0, 40.0]];
        let past_k = [
            [[0.2f32, 0.3], [-0.1, 0.6], [0.4, -0.5]],
            [[0.7, -0.3], [99.0, 99.0], [99.0, 99.0]], // garbage tail
        ];
        let past_v = [
            [[1.0f32, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[7.0, 8.0], [-99.0, -99.0], [-99.0, -99.0]],
        ];
        let seqlens = [3i32, 1];

        let flat2 = |x: &[[f32; 2]]| x.iter().flatten().copied().collect::<Vec<_>>();
        let flat3 = |x: &[[[f32; 2]; 3]]| x.iter().flatten().flatten().copied().collect::<Vec<_>>();
        let outs = run_composite(
            "com.microsoft.GroupQueryAttention",
            Attrs::new()
                .with("num_heads", AttrValue::Int(1))
                .with("kv_num_heads", AttrValue::Int(1)),
            vec![
                ("q", Tensor::from_f32(&flat2(&q), &[2, s, d]).unwrap()),
                ("k", Tensor::from_f32(&flat2(&k_new), &[2, s, d]).unwrap()),
                ("v", Tensor::from_f32(&flat2(&v_new), &[2, s, d]).unwrap()),
                (
                    "pk",
                    Tensor::from_f32(&flat3(&past_k), &[2, kv, p, d]).unwrap(),
                ),
                (
                    "pv",
                    Tensor::from_f32(&flat3(&past_v), &[2, kv, p, d]).unwrap(),
                ),
                (
                    "seqlens",
                    Tensor::new(
                        DataType::I32,
                        vec![2],
                        seqlens.iter().flat_map(|v| v.to_le_bytes()).collect(),
                    )
                    .unwrap(),
                ),
            ],
            vec![
                TensorType::of(DataType::F32, &[2, s as u64, d as u64]),
                TensorType::of(DataType::F32, &[2, kv as u64, (p + s) as u64, d as u64]),
                TensorType::of(DataType::F32, &[2, kv as u64, (p + s) as u64, d as u64]),
            ],
        );

        let scale = 1.0 / (d as f32).sqrt();
        let got = outs[0].1.to_f32().unwrap();
        for row in 0..2 {
            let valid_past = (seqlens[row] as usize + 1) - s;
            let mut keys: Vec<Vec<f32>> = past_k[row][..valid_past]
                .iter()
                .map(|k| k.to_vec())
                .collect();
            keys.push(k_new[row].to_vec());
            let mut vals: Vec<Vec<f32>> = past_v[row][..valid_past]
                .iter()
                .map(|v| v.to_vec())
                .collect();
            vals.push(v_new[row].to_vec());
            let expect = naive_attention(&[q[row].to_vec()], &keys, &vals, valid_past, scale, -1);
            assert_close(&got[row * d..(row + 1) * d], &expect[0], 1e-4);
        }

        // Row 1's present: valid past, then the new key, then zeros.
        let pk = outs[1].1.to_f32().unwrap();
        let row1 = &pk[(p + s) * d..2 * (p + s) * d];
        assert_close(&row1[..2], &[0.7, -0.3], 1e-6);
        assert_close(&row1[2..4], &[-0.2, 0.4], 1e-6);
        assert_close(&row1[4..], &[0.0; 4], 1e-6);
    }

    #[test]
    fn matmul_nbits_matches_float_matmul() {
        // K=8, N=2, block_size=4 → 2 blocks per row. Build quantized weights
        // by hand: w[n][k] = (q - 8) * scale.
        let (k_dim, n_dim, bs) = (8usize, 2usize, 4usize);
        let q_vals: Vec<u8> = (0..n_dim * k_dim).map(|i| (i % 16) as u8).collect();
        let scales: Vec<f32> = vec![0.5, 0.25, 1.0, 2.0]; // [N * n_blocks]
        // Pack q_vals as u4 (low nibble first).
        let mut packed = vec![0u8; (n_dim * k_dim).div_ceil(2)];
        for (i, &v) in q_vals.iter().enumerate() {
            packed[i / 2] |= (v & 0xF) << ((i % 2) * 4);
        }
        let n_blocks = k_dim / bs;
        let bq = Tensor::new(DataType::U4, vec![n_dim, n_blocks, bs], packed).unwrap();

        let a_vals: Vec<f32> = (0..k_dim).map(|i| (i as f32) * 0.1).collect();
        let outs = run_composite(
            "com.microsoft.MatMulNBits",
            Attrs::new()
                .with("K", AttrValue::Int(k_dim as i64))
                .with("N", AttrValue::Int(n_dim as i64))
                .with("bits", AttrValue::Int(4))
                .with("block_size", AttrValue::Int(bs as i64)),
            vec![
                ("a", Tensor::from_f32(&a_vals, &[1, k_dim]).unwrap()),
                ("b", bq),
                ("s", Tensor::from_f32(&scales, &[n_dim * n_blocks]).unwrap()),
            ],
            vec![TensorType::of(DataType::F32, &[1, n_dim as u64])],
        );

        // Reference float matmul with default zero point 8.
        let mut expect = vec![0.0f32; n_dim];
        for n in 0..n_dim {
            for kk in 0..k_dim {
                let block = n * n_blocks + kk / bs;
                let w = (q_vals[n * k_dim + kk] as f32 - 8.0) * scales[block];
                expect[n] += a_vals[kk] * w;
            }
        }
        assert_close(&outs[0].1.to_f32().unwrap(), &expect, 1e-4);
    }

    #[test]
    fn gqa_types_check_symbolically() {
        // Build GQA with symbolic S and T; inline; validate. This is the
        // shape-algebra stress test: present = [B, KV, T+S, D] must come out
        // canonically equal from both the declaration and the decomposition.
        let mut b = GraphBuilder::new();
        let s = b.sym("S");
        let t = b.sym("T");
        let f = DataType::F32;
        let q = b.input(
            "q",
            TensorType::new(f, SymbolicShape(vec![1u64.into(), s.clone(), 8u64.into()])),
        );
        let k = b.input(
            "k",
            TensorType::new(f, SymbolicShape(vec![1u64.into(), s.clone(), 4u64.into()])),
        );
        let v = b.input(
            "v",
            TensorType::new(f, SymbolicShape(vec![1u64.into(), s.clone(), 4u64.into()])),
        );
        let pk = b.input(
            "pk",
            TensorType::new(
                f,
                SymbolicShape(vec![1u64.into(), 1u64.into(), t.clone(), 4u64.into()]),
            ),
        );
        let pv = b.input(
            "pv",
            TensorType::new(
                f,
                SymbolicShape(vec![1u64.into(), 1u64.into(), t.clone(), 4u64.into()]),
            ),
        );
        let sl = b.input(
            "seqlens",
            TensorType::new(DataType::I32, SymbolicShape(vec![1u64.into()])),
        );
        let present_shape =
            SymbolicShape(vec![1u64.into(), 1u64.into(), t + s.clone(), 4u64.into()]);
        let outs = b
            .composite(
                "com.microsoft.GroupQueryAttention",
                Attrs::new()
                    .with("num_heads", AttrValue::Int(2))
                    .with("kv_num_heads", AttrValue::Int(1)),
                &[q, k, v, pk, pv, sl],
                vec![
                    TensorType::new(f, SymbolicShape(vec![1u64.into(), s, 8u64.into()])),
                    TensorType::new(f, present_shape.clone()),
                    TensorType::new(f, present_shape),
                ],
            )
            .unwrap();
        for (i, &o) in outs.iter().enumerate() {
            b.output(&format!("o{i}"), o);
        }
        let module = b.finish().unwrap();
        let module = inline_composites(module, &standard_decompositions(), &|_| false).unwrap();
        crate::validate::validate(&module).unwrap();
    }
}
