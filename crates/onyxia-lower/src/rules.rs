//! Lowering rules: one function per ONNX op type.
//!
//! Conventions:
//! - Rules normalize everything (negative axes, defaults, opset variants
//!   detected structurally) so the IR sees one canonical form.
//! - Shape-domain ops try [`LowerCtx::try_content`] first, so shape
//!   arithmetic evaluates at lowering instead of becoming runtime nodes.
//! - Ops with fused backend kernels are emitted as *composites*
//!   (`Softmax`, `Gelu`, `Trilu`, RMS-norm, and the `com.microsoft`
//!   attention/quantization ops); their portable decompositions live in
//!   [`onyxia_ir::decomp`].

use crate::{LowerCtx, LoweringRegistry, attrs, convert_proto_dtype, signed_const_of};
use onyxia_ir::graph::Origin;
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, ReduceOp, SliceSpec, UnaryOp};
use onyxia_ir::{AttrValue, DataType, DimExpr, Error, Result, TensorType, ValueId};

pub(crate) fn register_all(r: &mut LoweringRegistry) {
    // Element-wise binary (+ variadic Max/Min).
    r.register("", "Add", |c| binary(c, BinaryOp::Add));
    r.register("", "Sub", |c| binary(c, BinaryOp::Sub));
    r.register("", "Mul", |c| binary(c, BinaryOp::Mul));
    r.register("", "Div", |c| binary(c, BinaryOp::Div));
    r.register("", "Pow", pow);
    r.register("", "Max", |c| variadic(c, BinaryOp::Max));
    r.register("", "Min", |c| variadic(c, BinaryOp::Min));
    r.register("", "And", |c| binary(c, BinaryOp::And));
    r.register("", "Or", |c| binary(c, BinaryOp::Or));
    r.register("", "Xor", |c| binary(c, BinaryOp::Xor));

    // Comparisons.
    r.register("", "Equal", |c| compare(c, CmpOp::Eq));
    r.register("", "Greater", |c| compare(c, CmpOp::Gt));
    r.register("", "GreaterOrEqual", |c| compare(c, CmpOp::Ge));
    r.register("", "Less", |c| compare(c, CmpOp::Lt));
    r.register("", "LessOrEqual", |c| compare(c, CmpOp::Le));

    // Element-wise unary.
    r.register("", "Neg", |c| unary(c, UnaryOp::Neg));
    r.register("", "Abs", |c| unary(c, UnaryOp::Abs));
    r.register("", "Sqrt", |c| unary(c, UnaryOp::Sqrt));
    r.register("", "Exp", |c| unary(c, UnaryOp::Exp));
    r.register("", "Log", |c| unary(c, UnaryOp::Log));
    r.register("", "Sin", |c| unary(c, UnaryOp::Sin));
    r.register("", "Cos", |c| unary(c, UnaryOp::Cos));
    r.register("", "Tanh", |c| unary(c, UnaryOp::Tanh));
    r.register("", "Erf", |c| unary(c, UnaryOp::Erf));
    r.register("", "Floor", |c| unary(c, UnaryOp::Floor));
    r.register("", "Ceil", |c| unary(c, UnaryOp::Ceil));
    r.register("", "Not", |c| unary(c, UnaryOp::Not));

    // Structure.
    r.register("", "Cast", cast);
    r.register("", "Where", where_);
    r.register("", "MatMul", matmul);
    r.register("", "Reshape", reshape);
    r.register("", "Transpose", transpose);
    r.register("", "Expand", expand);
    r.register("", "Unsqueeze", unsqueeze);
    r.register("", "Squeeze", squeeze);
    r.register("", "Concat", concat);
    r.register("", "Slice", slice);
    r.register("", "Gather", gather);
    r.register("", "ScatterND", scatter_nd);
    r.register("", "Shape", shape);
    r.register("", "Range", range);
    r.register("", "ConstantOfShape", constant_of_shape);
    r.register("", "Identity", identity);

    // Reductions.
    r.register("", "ReduceSum", |c| reduce(c, ReduceOp::Sum));
    r.register("", "ReduceMean", |c| reduce(c, ReduceOp::Mean));
    r.register("", "ReduceMax", |c| reduce(c, ReduceOp::Max));
    r.register("", "ReduceMin", |c| reduce(c, ReduceOp::Min));
    r.register("", "ReduceProd", |c| reduce(c, ReduceOp::Prod));

    // Composites (fused-kernel candidates).
    r.register("", "Softmax", softmax);
    r.register("", "Gelu", gelu);
    r.register("", "Trilu", trilu);
    r.register("", "SimplifiedLayerNormalization", simplified_layer_norm);
    r.register("com.microsoft", "RotaryEmbedding", |c| {
        rotary(c, "com.microsoft.RotaryEmbedding")
    });
    r.register("com.microsoft", "GemmaRotaryEmbedding", |c| {
        rotary(c, "com.microsoft.GemmaRotaryEmbedding")
    });
    r.register(
        "com.microsoft",
        "GroupQueryAttention",
        group_query_attention,
    );
    r.register("com.microsoft", "MatMulNBits", matmul_nbits);
}

// ─────────────────────────── element-wise ──────────────────────────────

fn binary(ctx: &mut LowerCtx, op: BinaryOp) -> Result<()> {
    let prim = Prim::Binary(op);
    if matches!(
        op,
        BinaryOp::Add | BinaryOp::Sub | BinaryOp::Mul | BinaryOp::Div
    ) && ctx.try_content(&prim)?
    {
        return Ok(());
    }
    let (a, b) = (ctx.value(0)?, ctx.value(1)?);
    let out = ctx.emit(prim, &[a, b])?;
    ctx.set_value(0, out);
    Ok(())
}

/// Pow allows a different exponent dtype in ONNX; align it to the base.
fn pow(ctx: &mut LowerCtx) -> Result<()> {
    let (a, mut b) = (ctx.value(0)?, ctx.value(1)?);
    let base_dt = ctx.ty(a).dtype;
    if ctx.ty(b).dtype != base_dt {
        b = ctx.emit(Prim::Cast { to: base_dt }, &[b])?;
    }
    let out = ctx.emit(Prim::Binary(BinaryOp::Pow), &[a, b])?;
    ctx.set_value(0, out);
    Ok(())
}

/// ONNX Max/Min are variadic; fold left.
fn variadic(ctx: &mut LowerCtx, op: BinaryOp) -> Result<()> {
    let mut acc = ctx.value(0)?;
    for i in 1..ctx.num_inputs() {
        let rhs = ctx.value(i)?;
        acc = ctx.emit(Prim::Binary(op), &[acc, rhs])?;
    }
    ctx.set_value(0, acc);
    Ok(())
}

fn compare(ctx: &mut LowerCtx, op: CmpOp) -> Result<()> {
    let (a, b) = (ctx.value(0)?, ctx.value(1)?);
    let out = ctx.emit(Prim::Compare(op), &[a, b])?;
    ctx.set_value(0, out);
    Ok(())
}

fn unary(ctx: &mut LowerCtx, op: UnaryOp) -> Result<()> {
    let x = ctx.value(0)?;
    let out = ctx.emit(Prim::Unary(op), &[x])?;
    ctx.set_value(0, out);
    Ok(())
}

fn cast(ctx: &mut LowerCtx) -> Result<()> {
    let to = convert_proto_dtype(ctx.attr_i("to").ok_or_else(|| ctx.missing_attr("to"))?)?;
    let prim = Prim::Cast { to };
    if ctx.try_content(&prim)? {
        return Ok(());
    }
    let x = ctx.value(0)?;
    let out = ctx.emit(prim, &[x])?;
    ctx.set_value(0, out);
    Ok(())
}

fn where_(ctx: &mut LowerCtx) -> Result<()> {
    let (c, a, b) = (ctx.value(0)?, ctx.value(1)?, ctx.value(2)?);
    let out = ctx.emit(Prim::Select, &[c, a, b])?;
    ctx.set_value(0, out);
    Ok(())
}

// ───────────────────────────── structure ───────────────────────────────

/// MatMul with numpy rank-1 promotion.
fn matmul(ctx: &mut LowerCtx) -> Result<()> {
    let (mut a, mut b) = (ctx.value(0)?, ctx.value(1)?);
    let a1 = ctx.ty(a).shape.rank() == 1;
    let b1 = ctx.ty(b).shape.rank() == 1;
    if a1 {
        let k = ctx.ty(a).shape.dims()[0].clone();
        a = ctx.emit(
            Prim::Reshape {
                shape: vec![DimExpr::constant(1), k],
            },
            &[a],
        )?;
    }
    if b1 {
        let k = ctx.ty(b).shape.dims()[0].clone();
        b = ctx.emit(
            Prim::Reshape {
                shape: vec![k, DimExpr::constant(1)],
            },
            &[b],
        )?;
    }
    let mut out = ctx.emit(
        Prim::MatMul {
            trans_a: false,
            trans_b: false,
        },
        &[a, b],
    )?;
    if a1 || b1 {
        let dims = ctx.ty(out).shape.dims().to_vec();
        let squeezed: Vec<DimExpr> = dims
            .iter()
            .enumerate()
            .filter(|&(i, _)| !(a1 && i == dims.len() - 2 || b1 && i == dims.len() - 1))
            .map(|(_, d)| d.clone())
            .collect();
        out = ctx.emit(Prim::Reshape { shape: squeezed }, &[out])?;
    }
    ctx.set_value(0, out);
    Ok(())
}

fn reshape(ctx: &mut LowerCtx) -> Result<()> {
    let x = ctx.value(0)?;
    let in_dims = ctx.ty(x).shape.dims().to_vec();
    let target = ctx.content(1).ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}': Reshape target is not resolvable at compile time",
            ctx.node_name()
        ))
    })?;
    let allowzero = ctx.attr_i("allowzero").unwrap_or(0) != 0;

    // Resolve 0 (copy) and -1 (infer) entries.
    let mut dims: Vec<Option<DimExpr>> = Vec::with_capacity(target.elems.len());
    let mut inferred_at: Option<usize> = None;
    for (i, e) in target.elems.iter().enumerate() {
        match signed_const_of(e) {
            Some(0) if !allowzero => dims.push(Some(in_dims.get(i).cloned().ok_or_else(|| {
                Error::Shape(format!("Reshape dim 0 at {i} exceeds input rank"))
            })?)),
            Some(-1) => {
                if inferred_at.replace(i).is_some() {
                    return Err(Error::Shape("Reshape allows at most one -1".into()));
                }
                dims.push(None);
            }
            Some(v) if v < 0 => {
                return Err(Error::Shape(format!("invalid Reshape dim {v}")));
            }
            _ => dims.push(Some(e.clone())),
        }
    }
    if let Some(i) = inferred_at {
        let numel = onyxia_ir::SymbolicShape(in_dims).numel();
        let known = dims
            .iter()
            .flatten()
            .fold(DimExpr::constant(1), |acc, d| acc * d.clone());
        let inferred = numel.div_exact(&known).unwrap_or_else(|| {
            // Outside the exact-division fragment: late-bound, resolved
            // from the element count at run time.
            DimExpr::sym(ctx.builder().fresh_sym("reshape"))
        });
        dims[i] = Some(inferred);
    }
    let shape: Vec<DimExpr> = dims.into_iter().flatten().collect();

    let prim = Prim::Reshape { shape };
    if ctx.try_content(&prim)? {
        return Ok(());
    }
    let out = ctx.emit(prim, &[x])?;
    ctx.set_value(0, out);
    Ok(())
}

fn transpose(ctx: &mut LowerCtx) -> Result<()> {
    let x = ctx.value(0)?;
    let rank = ctx.ty(x).shape.rank();
    let perm: Vec<usize> = match ctx.attr_is("perm") {
        Some(p) => p
            .iter()
            .map(|&a| ctx.norm_axis(a, rank))
            .collect::<Result<_>>()?,
        None => (0..rank).rev().collect(),
    };
    let out = ctx.emit(Prim::Transpose { perm }, &[x])?;
    ctx.set_value(0, out);
    Ok(())
}

fn expand(ctx: &mut LowerCtx) -> Result<()> {
    let x = ctx.value(0)?;
    let target = ctx.content(1).ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}': Expand shape is not resolvable at compile time",
            ctx.node_name()
        ))
    })?;
    let out = ctx.emit(
        Prim::Broadcast {
            shape: target.elems,
        },
        &[x],
    )?;
    ctx.set_value(0, out);
    Ok(())
}

/// Axes for Unsqueeze/Squeeze/Reduce: input takes precedence (opset 13+),
/// attribute otherwise.
fn axes_of(ctx: &LowerCtx, input_idx: usize) -> Option<Vec<i64>> {
    if ctx.has_input(input_idx) {
        ctx.const_ints(input_idx)
    } else {
        ctx.attr_is("axes")
    }
}

fn unsqueeze(ctx: &mut LowerCtx) -> Result<()> {
    let in_dims: Vec<DimExpr> = match ctx.content(0) {
        // Shape-domain input: rank comes from the content.
        Some(c) => c
            .shape
            .iter()
            .map(|&d| DimExpr::constant(d as u64))
            .collect(),
        None => ctx.ty(ctx_value_peek(ctx, 0)?).shape.dims().to_vec(),
    };
    let axes = axes_of(ctx, 1)
        .ok_or_else(|| Error::Unsupported("Unsqueeze axes must be constant".into()))?;
    let out_rank = in_dims.len() + axes.len();
    let mut norm: Vec<usize> = axes
        .iter()
        .map(|&a| ctx.norm_axis(a, out_rank))
        .collect::<Result<_>>()?;
    norm.sort_unstable();

    let mut dims = in_dims.clone();
    // In-dims here are only used for rank bookkeeping on the content path.
    let mut target: Vec<DimExpr> = Vec::with_capacity(out_rank);
    let mut src = dims.drain(..);
    for i in 0..out_rank {
        if norm.contains(&i) {
            target.push(DimExpr::constant(1));
        } else {
            target.push(src.next().expect("rank bookkeeping"));
        }
    }
    reshape_to(ctx, target)
}

fn squeeze(ctx: &mut LowerCtx) -> Result<()> {
    let x_dims: Vec<DimExpr> = match ctx.content(0) {
        Some(c) => c
            .shape
            .iter()
            .map(|&d| DimExpr::constant(d as u64))
            .collect(),
        None => ctx.ty(ctx_value_peek(ctx, 0)?).shape.dims().to_vec(),
    };
    let rank = x_dims.len();
    let drop: Vec<usize> = match axes_of(ctx, 1) {
        Some(axes) => axes
            .iter()
            .map(|&a| ctx.norm_axis(a, rank))
            .collect::<Result<_>>()?,
        None => x_dims
            .iter()
            .enumerate()
            .filter(|(_, d)| d.as_const() == Some(1))
            .map(|(i, _)| i)
            .collect(),
    };
    let target: Vec<DimExpr> = x_dims
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !drop.contains(i))
        .map(|(_, d)| d)
        .collect();
    reshape_to(ctx, target)
}

/// Shared tail of Unsqueeze/Squeeze: reshape (content-aware).
fn reshape_to(ctx: &mut LowerCtx, shape: Vec<DimExpr>) -> Result<()> {
    let prim = Prim::Reshape { shape };
    if ctx.try_content(&prim)? {
        return Ok(());
    }
    let x = ctx.value(0)?;
    let out = ctx.emit(prim, &[x])?;
    ctx.set_value(0, out);
    Ok(())
}

/// Peek input `i`'s ValueId without materializing content (errors if it's
/// pure content — callers use this only on paths where a value must exist).
fn ctx_value_peek(ctx: &LowerCtx, i: usize) -> Result<ValueId> {
    match ctx.peek(i)? {
        crate::Lowered::Value(v) => Ok(*v),
        crate::Lowered::Content(_) => Err(Error::Unsupported(
            "expected a runtime value, found a compile-time shape value".into(),
        )),
    }
}

fn concat(ctx: &mut LowerCtx) -> Result<()> {
    let axis_attr = ctx.attr_i("axis").ok_or_else(|| ctx.missing_attr("axis"))?;
    // Rank for normalization: from the first runtime input, else 1
    // (shape-vector concatenation).
    let rank = (0..ctx.num_inputs())
        .find_map(|i| match ctx.peek(i).ok()? {
            crate::Lowered::Value(v) => Some(ctx.ty(*v).shape.rank()),
            crate::Lowered::Content(_) => None,
        })
        .unwrap_or(1);
    let axis = ctx.norm_axis(axis_attr, rank)?;
    let prim = Prim::Concat { axis };
    if ctx.try_content(&prim)? {
        return Ok(());
    }
    let inputs: Vec<ValueId> = (0..ctx.num_inputs())
        .map(|i| ctx.value(i))
        .collect::<Result<_>>()?;
    let out = ctx.emit(prim, &inputs)?;
    ctx.set_value(0, out);
    Ok(())
}

fn slice(ctx: &mut LowerCtx) -> Result<()> {
    // Bounds may be symbolic (shape-chain results); axes/steps must be
    // constant.
    let starts = ctx.content(1).ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}': Slice starts not resolvable at compile time",
            ctx.node_name()
        ))
    })?;
    let ends = ctx.content(2).ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}': Slice ends not resolvable at compile time",
            ctx.node_name()
        ))
    })?;
    let n = starts.elems.len();
    let data_dims: Vec<DimExpr> = match ctx.content(0) {
        Some(c) => c
            .shape
            .iter()
            .map(|&d| DimExpr::constant(d as u64))
            .collect(),
        None => ctx.ty(ctx_value_peek(ctx, 0)?).shape.dims().to_vec(),
    };
    let rank = data_dims.len();
    let axes: Vec<usize> = match if ctx.has_input(3) {
        ctx.const_ints(3)
    } else {
        None
    } {
        Some(a) => a
            .iter()
            .map(|&v| ctx.norm_axis(v, rank))
            .collect::<Result<_>>()?,
        None => (0..n).collect(),
    };
    let steps: Vec<i64> = match if ctx.has_input(4) {
        ctx.const_ints(4)
    } else {
        None
    } {
        Some(s) => s,
        None => vec![1; n],
    };

    let mut specs = Vec::with_capacity(n);
    for i in 0..n {
        let dim = &data_dims[axes[i]];
        let norm = |bound: &DimExpr, is_end: bool| -> DimExpr {
            match signed_const_of(bound) {
                // Huge sentinel (INT64_MAX or INT32_MAX): "to the end".
                Some(v) if is_end && v >= i32::MAX as i64 => dim.clone(),
                // Negative: from the end.
                Some(v) if v < 0 => dim.clone() - DimExpr::constant(v.unsigned_abs()),
                // Clamp constants against constant dims.
                Some(v) => match dim.as_const() {
                    Some(d) => DimExpr::constant((v as u64).min(d)),
                    None => DimExpr::constant(v as u64),
                },
                None => bound.clone(),
            }
        };
        specs.push(SliceSpec {
            axis: axes[i],
            start: norm(&starts.elems[i], false),
            end: norm(&ends.elems[i], true),
            step: steps[i],
        });
    }

    let prim = Prim::Slice { specs };
    if ctx.try_content(&prim)? {
        return Ok(());
    }
    let x = ctx.value(0)?;
    let out = ctx.emit(prim, &[x])?;
    ctx.set_value(0, out);
    Ok(())
}

fn gather(ctx: &mut LowerCtx) -> Result<()> {
    let axis_attr = ctx.attr_i("axis").unwrap_or(0);
    let rank = match ctx.peek(0)? {
        crate::Lowered::Value(v) => ctx.ty(*v).shape.rank(),
        crate::Lowered::Content(_) => 1,
    };
    let axis = ctx.norm_axis(axis_attr, rank)?;
    let prim = Prim::Gather { axis };
    if ctx.try_content(&prim)? {
        return Ok(());
    }
    let data = ctx.value(0)?;
    let mut indices = ctx.value(1)?;
    if ctx.ty(indices).dtype == DataType::I32 {
        indices = ctx.emit(Prim::Cast { to: DataType::I64 }, &[indices])?;
    }
    let out = ctx.emit(prim, &[data, indices])?;
    ctx.set_value(0, out);
    Ok(())
}

fn scatter_nd(ctx: &mut LowerCtx) -> Result<()> {
    if let Some(mode) = ctx.attr_s("reduction") {
        if mode != "none" {
            return Err(Error::Unsupported(format!("ScatterND reduction='{mode}'")));
        }
    }
    let (data, indices, updates) = (ctx.value(0)?, ctx.value(1)?, ctx.value(2)?);
    let out = ctx.emit(Prim::Scatter, &[data, indices, updates])?;
    ctx.set_value(0, out);
    Ok(())
}

/// Shape enters the compile-time value domain: its "output" is the input's
/// symbolic dims, never a runtime tensor.
fn shape(ctx: &mut LowerCtx) -> Result<()> {
    let dims: Vec<DimExpr> = match ctx.peek(0)? {
        crate::Lowered::Value(v) => ctx.ty(*v).shape.dims().to_vec(),
        crate::Lowered::Content(c) => c
            .shape
            .iter()
            .map(|&d| DimExpr::constant(d as u64))
            .collect(),
    };
    let rank = dims.len();
    let start = ctx
        .attr_i("start")
        .map(|v| clamp_shape_bound(v, rank))
        .unwrap_or(0);
    let end = ctx
        .attr_i("end")
        .map(|v| clamp_shape_bound(v, rank))
        .unwrap_or(rank);
    let content = onyxia_ir::graph::SymbolicContent::vector(dims[start.min(end)..end].to_vec());
    ctx.set_content(0, content);
    Ok(())
}

fn clamp_shape_bound(v: i64, rank: usize) -> usize {
    let v = if v < 0 { v + rank as i64 } else { v };
    v.clamp(0, rank as i64) as usize
}

/// Range lowers to `Iota` plus element-wise arithmetic. Start and delta
/// must be constant; the limit may be symbolic when delta is ±1-exact.
fn range(ctx: &mut LowerCtx) -> Result<()> {
    let get_scalar = |i: usize| -> Result<DimExpr> {
        ctx.content(i)
            .and_then(|c| c.elems.into_iter().next())
            .ok_or_else(|| {
                Error::Unsupported(format!(
                    "node '{}': Range bounds not resolvable at compile time",
                    ctx.node_name()
                ))
            })
    };
    let start_e = get_scalar(0)?;
    let limit_e = get_scalar(1)?;
    let delta_e = get_scalar(2)?;
    let start = signed_const_of(&start_e)
        .ok_or_else(|| Error::Unsupported("Range with symbolic start".into()))?;
    let delta = signed_const_of(&delta_e)
        .ok_or_else(|| Error::Unsupported("Range with symbolic delta".into()))?;
    if delta <= 0 {
        return Err(Error::Unsupported(format!("Range with delta {delta}")));
    }
    let span = limit_e - start_e;
    let len = span
        .div_exact(&DimExpr::constant(delta as u64))
        .ok_or_else(|| {
            Error::Unsupported("Range length is not exactly divisible by delta".into())
        })?;
    let mut out = ctx.builder().iota(len, DataType::I64)?;
    if delta != 1 {
        let d = ctx.builder().const_i64(&[delta], &[])?;
        out = ctx.emit(Prim::Binary(BinaryOp::Mul), &[out, d])?;
    }
    if start != 0 {
        let s = ctx.builder().const_i64(&[start], &[])?;
        out = ctx.emit(Prim::Binary(BinaryOp::Add), &[out, s])?;
    }
    ctx.set_value(0, out);
    Ok(())
}

/// ConstantOfShape = Broadcast(scalar, shape).
fn constant_of_shape(ctx: &mut LowerCtx) -> Result<()> {
    let target = ctx.content(0).ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}': ConstantOfShape shape not resolvable at compile time",
            ctx.node_name()
        ))
    })?;
    let (ty, bytes): (TensorType, Vec<u8>) = match ctx.attr_tensor("value") {
        // Per the ONNX spec, the default is a single f32 zero.
        None => (
            TensorType::of(DataType::F32, &[]),
            0f32.to_le_bytes().to_vec(),
        ),
        Some(t) => (
            TensorType::of(crate::convert_dtype(t.dtype), &[]),
            t.data.clone(),
        ),
    };
    let scalar = ctx.builder().constant(ty, bytes)?;
    let out = ctx.emit(
        Prim::Broadcast {
            shape: target.elems,
        },
        &[scalar],
    )?;
    ctx.set_value(0, out);
    Ok(())
}

fn identity(ctx: &mut LowerCtx) -> Result<()> {
    let lowered = ctx.peek(0)?.clone();
    ctx.set_lowered(0, lowered);
    Ok(())
}

fn reduce(ctx: &mut LowerCtx, op: ReduceOp) -> Result<()> {
    // Content path first (e.g. ReduceProd over a shape vector = numel).
    let keepdims = ctx.attr_i("keepdims").unwrap_or(1) != 0;
    if ctx.content(0).is_some() {
        let axes = axes_of(ctx, 1).unwrap_or_else(|| vec![0]);
        if axes == [0] {
            let prim = Prim::Reduce {
                op,
                axes: vec![0],
                keepdims,
            };
            if ctx.try_content(&prim)? {
                return Ok(());
            }
        }
    }

    let x = ctx.value(0)?;
    let rank = ctx.ty(x).shape.rank();
    let axes_raw = axes_of(ctx, 1);
    let noop_empty = ctx.attr_i("noop_with_empty_axes").unwrap_or(0) != 0;
    let axes: Vec<usize> = match axes_raw {
        Some(a) if !a.is_empty() => {
            let mut v: Vec<usize> = a
                .iter()
                .map(|&x| ctx.norm_axis(x, rank))
                .collect::<Result<_>>()?;
            v.sort_unstable();
            v.dedup();
            v
        }
        _ if noop_empty => {
            ctx.set_value(0, x);
            return Ok(());
        }
        _ => (0..rank).collect(),
    };
    let out = ctx.emit(Prim::Reduce { op, axes, keepdims }, &[x])?;
    ctx.set_value(0, out);
    Ok(())
}

// ─────────────────────────── composites ────────────────────────────────

fn softmax(ctx: &mut LowerCtx) -> Result<()> {
    let x = ctx.value(0)?;
    let rank = ctx.ty(x).shape.rank();
    let axis = ctx.norm_axis(ctx.attr_i("axis").unwrap_or(-1), rank)?;
    let ty = ctx.ty(x).clone();
    let outs = ctx.builder().composite(
        "Softmax",
        attrs(vec![("axis", AttrValue::Int(axis as i64))]),
        &[x],
        vec![ty],
    )?;
    ctx.set_value(0, outs[0]);
    Ok(())
}

fn gelu(ctx: &mut LowerCtx) -> Result<()> {
    let x = ctx.value(0)?;
    let approx = ctx.attr_s("approximate").unwrap_or("none").to_string();
    let ty = ctx.ty(x).clone();
    let outs = ctx.builder().composite(
        "Gelu",
        attrs(vec![("approximate", AttrValue::Str(approx))]),
        &[x],
        vec![ty],
    )?;
    ctx.set_value(0, outs[0]);
    Ok(())
}

fn trilu(ctx: &mut LowerCtx) -> Result<()> {
    let x = ctx.value(0)?;
    let upper = ctx.attr_i("upper").unwrap_or(1);
    let k = if ctx.has_input(1) {
        ctx.const_ints(1)
            .and_then(|v| v.first().copied())
            .ok_or_else(|| Error::Unsupported("Trilu k must be constant".into()))?
    } else {
        0
    };
    let ty = ctx.ty(x).clone();
    let outs = ctx.builder().composite(
        "Trilu",
        attrs(vec![
            ("upper", AttrValue::Int(upper)),
            ("k", AttrValue::Int(k)),
        ]),
        &[x],
        vec![ty],
    )?;
    ctx.set_value(0, outs[0]);
    Ok(())
}

fn simplified_layer_norm(ctx: &mut LowerCtx) -> Result<()> {
    let (x, w) = (ctx.value(0)?, ctx.value(1)?);
    let epsilon = ctx.attr_f("epsilon").unwrap_or(1e-5);
    let ty = ctx.ty(x).clone();
    let outs = ctx.builder().composite(
        "SimplifiedLayerNormalization",
        attrs(vec![("epsilon", AttrValue::Float(epsilon as f64))]),
        &[x, w],
        vec![ty],
    )?;
    ctx.set_value(0, outs[0]);
    Ok(())
}

fn rotary(ctx: &mut LowerCtx, name: &str) -> Result<()> {
    let inputs: Vec<ValueId> = (0..4).map(|i| ctx.value(i)).collect::<Result<_>>()?;
    let a = attrs(vec![
        (
            "interleaved",
            AttrValue::Int(ctx.attr_i("interleaved").unwrap_or(0)),
        ),
        (
            "num_heads",
            AttrValue::Int(ctx.attr_i("num_heads").unwrap_or(0).max(0)),
        ),
        (
            "rotary_embedding_dim",
            AttrValue::Int(ctx.attr_i("rotary_embedding_dim").unwrap_or(0)),
        ),
    ]);
    let ty = ctx.ty(inputs[0]).clone();
    let outs = ctx.builder().composite(name, a, &inputs, vec![ty])?;
    ctx.set_value(0, outs[0]);
    Ok(())
}

fn group_query_attention(ctx: &mut LowerCtx) -> Result<()> {
    if ctx.attr_i("do_rotary").unwrap_or(0) != 0 {
        return Err(Error::Unsupported(
            "GroupQueryAttention with do_rotary=1".into(),
        ));
    }
    if ctx.attr_f("softcap").unwrap_or(0.0) != 0.0 {
        return Err(Error::Unsupported(
            "GroupQueryAttention with softcap".into(),
        ));
    }
    // Only q/k/v and the past KV enter the composite. Inputs 5–6
    // (seqlens_k, total_sequence_length) are redundant under symbolic
    // dims — real exports often compute total_sequence_length via shape
    // arithmetic, which has no runtime representation here. A backend
    // kernel that wants them can rematerialize from bound symbols at
    // prepare time.
    if ctx.num_inputs() < 5 || !(0..5).all(|i| ctx.has_input(i)) {
        return Err(Error::Unsupported(
            "GroupQueryAttention without explicit past KV inputs".into(),
        ));
    }
    let inputs: Vec<ValueId> = (0..5).map(|i| ctx.value(i)).collect::<Result<_>>()?;
    let num_heads = ctx
        .attr_i("num_heads")
        .ok_or_else(|| ctx.missing_attr("num_heads"))?;
    let kv_heads = ctx
        .attr_i("kv_num_heads")
        .ok_or_else(|| ctx.missing_attr("kv_num_heads"))?;
    let a = attrs(vec![
        ("num_heads", AttrValue::Int(num_heads)),
        ("kv_num_heads", AttrValue::Int(kv_heads)),
        (
            "scale",
            AttrValue::Float(ctx.attr_f("scale").unwrap_or(0.0) as f64),
        ),
        (
            "local_window_size",
            AttrValue::Int(ctx.attr_i("local_window_size").unwrap_or(-1)),
        ),
    ]);

    let q_ty = ctx.ty(inputs[0]).clone();
    let past_ty = ctx.ty(inputs[3]).clone();
    let seq = q_ty.shape.dims()[1].clone();
    let mut present_dims = past_ty.shape.dims().to_vec();
    present_dims[2] = present_dims[2].clone() + seq;
    let present_ty = TensorType::new(past_ty.dtype, onyxia_ir::SymbolicShape(present_dims));

    let outs = ctx.builder().composite(
        "com.microsoft.GroupQueryAttention",
        a,
        &inputs,
        vec![q_ty, present_ty.clone(), present_ty],
    )?;
    for (i, &o) in outs.iter().enumerate() {
        ctx.set_value(i, o);
    }
    Ok(())
}

fn matmul_nbits(ctx: &mut LowerCtx) -> Result<()> {
    let bits = ctx.attr_i("bits").unwrap_or(4);
    if bits != 4 {
        return Err(Error::Unsupported(format!("MatMulNBits bits={bits}")));
    }
    let k = ctx.attr_i("K").ok_or_else(|| ctx.missing_attr("K"))? as u64;
    let n = ctx.attr_i("N").ok_or_else(|| ctx.missing_attr("N"))? as u64;
    let block_size = ctx
        .attr_i("block_size")
        .ok_or_else(|| ctx.missing_attr("block_size"))? as u64;
    let n_blocks = k.div_ceil(block_size);

    let a = ctx.value(0)?;
    let b = ctx.value(1)?;
    // Reinterpret the packed byte blob as logical U4 [N, n_blocks, block],
    // in place — same bytes, honest logical type.
    reinterpret_const(
        ctx,
        b,
        TensorType::of(DataType::U4, &[n, n_blocks, block_size]),
    )?;
    let scales = ctx.value(2)?;
    let mut inputs = vec![a, b, scales];
    if ctx.has_input(3) {
        let zp = ctx.value(3)?;
        reinterpret_const(ctx, zp, TensorType::of(DataType::U4, &[n, n_blocks])).map_err(|e| {
            Error::Unsupported(format!(
                "MatMulNBits zero_points layout not reinterpretable \
                     (odd n_blocks rows are per-row padded): {e}"
            ))
        })?;
        inputs.push(zp);
    }

    let a_dims = ctx.ty(a).shape.dims().to_vec();
    let mut out_dims = a_dims[..a_dims.len() - 1].to_vec();
    out_dims.push(DimExpr::constant(n));
    let out_ty = TensorType::new(ctx.ty(a).dtype, onyxia_ir::SymbolicShape(out_dims));

    let attrs_ = attrs(vec![
        ("K", AttrValue::Int(k as i64)),
        ("N", AttrValue::Int(n as i64)),
        ("bits", AttrValue::Int(bits)),
        ("block_size", AttrValue::Int(block_size as i64)),
    ]);
    let outs =
        ctx.builder()
            .composite("com.microsoft.MatMulNBits", attrs_, &inputs, vec![out_ty])?;
    ctx.set_value(0, outs[0]);
    Ok(())
}

/// Rewrap a constant value's type in place (zero-copy).
fn reinterpret_const(ctx: &mut LowerCtx, v: ValueId, ty: TensorType) -> Result<()> {
    let module = ctx.builder().module_mut();
    let Origin::Const(cid) = module.value(v).origin else {
        return Err(Error::Unsupported(
            "quantized weights must be constant initializers".into(),
        ));
    };
    module.consts.reinterpret(cid, ty.clone())?;
    module.value_mut(v).ty = ty;
    Ok(())
}
