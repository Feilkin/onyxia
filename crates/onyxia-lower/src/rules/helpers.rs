//! Shared building blocks for lowering rules: typed scalars, constant
//! reads, shape bookkeeping, and the index-arithmetic idioms (iota along an
//! axis, linear-index gathers) most structural ops are built from.

use crate::{LowerCtx, Lowered};
use onyxia_ir::prim::{BinaryOp, CmpOp, Prim, ReduceOp, SliceSpec, UnaryOp};
use onyxia_ir::{DataType, DimExpr, Error, Result, TensorType, ValueId};

pub(crate) use super::{const_values_f64, typed_scalar};

// ───────────────────────────── inputs ──────────────────────────────────

/// Input `i` as a runtime value (materializing content).
pub(crate) fn val(ctx: &mut LowerCtx, i: usize) -> Result<ValueId> {
    ctx.value(i)
}

/// Optional input `i` as a runtime value.
pub(crate) fn opt_val(ctx: &mut LowerCtx, i: usize) -> Result<Option<ValueId>> {
    if ctx.has_input(i) {
        Ok(Some(ctx.value(i)?))
    } else {
        Ok(None)
    }
}

/// Input `i` as constant i64s (any integer dtype), if it is constant.
pub(crate) fn const_ints(ctx: &LowerCtx, i: usize) -> Option<Vec<i64>> {
    if let Some(v) = ctx.const_ints(i) {
        return Some(v);
    }
    let v = match ctx.peek(i).ok()? {
        Lowered::Value(v) => *v,
        Lowered::Content(_) => return None,
    };
    let vals = const_values_f64(ctx, v)?;
    Some(vals.iter().map(|&x| x as i64).collect())
}

/// Required constant integer input `i`.
pub(crate) fn require_const_ints(ctx: &LowerCtx, i: usize, what: &str) -> Result<Vec<i64>> {
    const_ints(ctx, i).ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}': {what} must be constant",
            ctx.node_name()
        ))
    })
}

/// Input `i` as constant f64s (any numeric dtype), if it is constant.
pub(crate) fn const_floats(ctx: &LowerCtx, i: usize) -> Option<Vec<f64>> {
    if let Some(v) = ctx.const_ints(i) {
        return Some(v.iter().map(|&x| x as f64).collect());
    }
    let v = match ctx.peek(i).ok()? {
        Lowered::Value(v) => *v,
        Lowered::Content(_) => return None,
    };
    const_values_f64(ctx, v)
}

// ───────────────────────────── shapes ──────────────────────────────────

pub(crate) fn dims(ctx: &LowerCtx, v: ValueId) -> Vec<DimExpr> {
    ctx.ty(v).shape.dims().to_vec()
}

pub(crate) fn rank(ctx: &LowerCtx, v: ValueId) -> usize {
    ctx.ty(v).shape.rank()
}

pub(crate) fn dtype(ctx: &LowerCtx, v: ValueId) -> DataType {
    ctx.ty(v).dtype
}

/// Static dims of a value, or an error naming the op.
pub(crate) fn static_dims(ctx: &LowerCtx, v: ValueId, what: &str) -> Result<Vec<u64>> {
    ctx.ty(v).shape.as_static().ok_or_else(|| {
        Error::Unsupported(format!(
            "node '{}': {what} requires static shapes, got {}",
            ctx.node_name(),
            ctx.ty(v).shape
        ))
    })
}

pub(crate) fn c(v: u64) -> DimExpr {
    DimExpr::constant(v)
}

pub(crate) fn prod(dims: &[DimExpr]) -> DimExpr {
    dims.iter().cloned().fold(c(1), |a, b| a * b)
}

// ───────────────────────────── emitters ────────────────────────────────

pub(crate) fn unary(ctx: &mut LowerCtx, op: UnaryOp, x: ValueId) -> Result<ValueId> {
    ctx.emit(Prim::Unary(op), &[x])
}

pub(crate) fn binary(ctx: &mut LowerCtx, op: BinaryOp, a: ValueId, b: ValueId) -> Result<ValueId> {
    ctx.emit(Prim::Binary(op), &[a, b])
}

pub(crate) fn add(ctx: &mut LowerCtx, a: ValueId, b: ValueId) -> Result<ValueId> {
    binary(ctx, BinaryOp::Add, a, b)
}
pub(crate) fn sub(ctx: &mut LowerCtx, a: ValueId, b: ValueId) -> Result<ValueId> {
    binary(ctx, BinaryOp::Sub, a, b)
}
pub(crate) fn mul(ctx: &mut LowerCtx, a: ValueId, b: ValueId) -> Result<ValueId> {
    binary(ctx, BinaryOp::Mul, a, b)
}
pub(crate) fn div(ctx: &mut LowerCtx, a: ValueId, b: ValueId) -> Result<ValueId> {
    binary(ctx, BinaryOp::Div, a, b)
}
pub(crate) fn max(ctx: &mut LowerCtx, a: ValueId, b: ValueId) -> Result<ValueId> {
    binary(ctx, BinaryOp::Max, a, b)
}
pub(crate) fn min(ctx: &mut LowerCtx, a: ValueId, b: ValueId) -> Result<ValueId> {
    binary(ctx, BinaryOp::Min, a, b)
}

pub(crate) fn cmp(ctx: &mut LowerCtx, op: CmpOp, a: ValueId, b: ValueId) -> Result<ValueId> {
    ctx.emit(Prim::Compare(op), &[a, b])
}

pub(crate) fn select(ctx: &mut LowerCtx, cond: ValueId, a: ValueId, b: ValueId) -> Result<ValueId> {
    ctx.emit(Prim::Select, &[cond, a, b])
}

pub(crate) fn cast(ctx: &mut LowerCtx, x: ValueId, to: DataType) -> Result<ValueId> {
    if dtype(ctx, x) == to {
        return Ok(x);
    }
    ctx.emit(Prim::Cast { to }, &[x])
}

pub(crate) fn reshape(ctx: &mut LowerCtx, x: ValueId, shape: Vec<DimExpr>) -> Result<ValueId> {
    if dims(ctx, x) == shape {
        return Ok(x);
    }
    ctx.emit(Prim::Reshape { shape }, &[x])
}

pub(crate) fn transpose(ctx: &mut LowerCtx, x: ValueId, perm: &[usize]) -> Result<ValueId> {
    if perm.iter().enumerate().all(|(i, &p)| i == p) {
        return Ok(x);
    }
    ctx.emit(
        Prim::Transpose {
            perm: perm.to_vec(),
        },
        &[x],
    )
}

pub(crate) fn broadcast(ctx: &mut LowerCtx, x: ValueId, shape: Vec<DimExpr>) -> Result<ValueId> {
    if dims(ctx, x) == shape {
        return Ok(x);
    }
    ctx.emit(Prim::Broadcast { shape }, &[x])
}

pub(crate) fn concat(ctx: &mut LowerCtx, inputs: &[ValueId], axis: usize) -> Result<ValueId> {
    if inputs.len() == 1 {
        return Ok(inputs[0]);
    }
    ctx.emit(Prim::Concat { axis }, inputs)
}

pub(crate) fn reduce(
    ctx: &mut LowerCtx,
    op: ReduceOp,
    x: ValueId,
    axes: &[usize],
    keepdims: bool,
) -> Result<ValueId> {
    if axes.is_empty() {
        return Ok(x);
    }
    ctx.emit(
        Prim::Reduce {
            op,
            axes: axes.to_vec(),
            keepdims,
        },
        &[x],
    )
}

pub(crate) fn matmul(ctx: &mut LowerCtx, a: ValueId, b: ValueId) -> Result<ValueId> {
    ctx.emit(
        Prim::MatMul {
            trans_a: false,
            trans_b: false,
        },
        &[a, b],
    )
}

/// Slice one axis `[start, end)` with step 1 (constant bounds).
pub(crate) fn slice_axis(
    ctx: &mut LowerCtx,
    x: ValueId,
    axis: usize,
    start: u64,
    end: u64,
) -> Result<ValueId> {
    slice_axis_dyn(ctx, x, axis, c(start), c(end))
}

pub(crate) fn slice_axis_dyn(
    ctx: &mut LowerCtx,
    x: ValueId,
    axis: usize,
    start: DimExpr,
    end: DimExpr,
) -> Result<ValueId> {
    let d = dims(ctx, x);
    if start == c(0) && end == d[axis] {
        return Ok(x);
    }
    ctx.emit(
        Prim::Slice {
            specs: vec![SliceSpec {
                axis,
                start,
                end,
                step: 1,
            }],
        },
        &[x],
    )
}

/// Reverse one axis (constant length).
pub(crate) fn flip(ctx: &mut LowerCtx, x: ValueId, axis: usize) -> Result<ValueId> {
    let n = dims(ctx, x)[axis]
        .as_const()
        .ok_or_else(|| Error::Unsupported("flip of a symbolic axis".into()))?;
    if n <= 1 {
        return Ok(x);
    }
    ctx.emit(
        Prim::Slice {
            specs: vec![SliceSpec {
                axis,
                start: c(n - 1),
                end: DimExpr::constant(0) - DimExpr::constant(1),
                step: -1,
            }],
        },
        &[x],
    )
}

/// Insert a size-1 axis at `axis`.
pub(crate) fn unsqueeze(ctx: &mut LowerCtx, x: ValueId, axis: usize) -> Result<ValueId> {
    let mut d = dims(ctx, x);
    d.insert(axis, c(1));
    reshape(ctx, x, d)
}

/// Remove size-1 axis `axis`.
pub(crate) fn squeeze(ctx: &mut LowerCtx, x: ValueId, axis: usize) -> Result<ValueId> {
    let mut d = dims(ctx, x);
    d.remove(axis);
    reshape(ctx, x, d)
}

/// A rank-`rank` constant of `dt`, all dims 1 (broadcasts anywhere).
pub(crate) fn scalar(ctx: &mut LowerCtx, dt: DataType, v: f64) -> Result<ValueId> {
    typed_scalar(ctx, dt, v)
}

/// Constant i64 vector.
pub(crate) fn const_i64(ctx: &mut LowerCtx, vals: &[i64], dims: &[u64]) -> Result<ValueId> {
    ctx.builder().const_i64(vals, dims)
}

/// Constant tensor of `dt` from f64 values.
pub(crate) fn const_typed(
    ctx: &mut LowerCtx,
    dt: DataType,
    vals: &[f64],
    dims: &[u64],
) -> Result<ValueId> {
    let bytes: Vec<u8> = match dt {
        DataType::F32 => vals
            .iter()
            .flat_map(|&v| (v as f32).to_le_bytes())
            .collect(),
        DataType::F16 => vals
            .iter()
            .flat_map(|&v| half::f16::from_f64(v).to_le_bytes())
            .collect(),
        DataType::I64 => vals
            .iter()
            .flat_map(|&v| (v as i64).to_le_bytes())
            .collect(),
        DataType::I32 => vals
            .iter()
            .flat_map(|&v| (v as i32).to_le_bytes())
            .collect(),
        DataType::U32 => vals
            .iter()
            .flat_map(|&v| (v as u32).to_le_bytes())
            .collect(),
        DataType::U8 => vals.iter().map(|&v| v as u8).collect(),
        DataType::I8 => vals.iter().map(|&v| v as i8 as u8).collect(),
        DataType::Bool => vals.iter().map(|&v| (v != 0.0) as u8).collect(),
        other => {
            return Err(Error::Unsupported(format!("constant of dtype {other}")));
        }
    };
    ctx.builder().constant(TensorType::of(dt, dims), bytes)
}

/// `iota(len)` as `dtype`, reshaped to rank `rank` with the ramp along
/// `axis` (other dims 1) — the broadcastable coordinate tensor.
pub(crate) fn iota_along(
    ctx: &mut LowerCtx,
    len: DimExpr,
    rank: usize,
    axis: usize,
    dt: DataType,
) -> Result<ValueId> {
    let v = ctx.builder().iota(len.clone(), DataType::I64)?;
    let v = cast(ctx, v, dt)?;
    let mut shape = vec![c(1); rank];
    shape[axis] = len;
    reshape(ctx, v, shape)
}

/// Normalize possibly-negative indices against `dim`:
/// `select(idx < 0, idx + dim, idx)`.
pub(crate) fn wrap_negative(ctx: &mut LowerCtx, idx: ValueId, dim: DimExpr) -> Result<ValueId> {
    let zero = scalar(ctx, DataType::I64, 0.0)?;
    let d = dim_value(ctx, dim)?;
    let neg = cmp(ctx, CmpOp::Lt, idx, zero)?;
    let wrapped = add(ctx, idx, d)?;
    select(ctx, neg, wrapped, idx)
}

/// A dim expression as a rank-0 i64 value (constant or `DimValues`).
pub(crate) fn dim_value(ctx: &mut LowerCtx, d: DimExpr) -> Result<ValueId> {
    if let Some(v) = d.as_signed_const() {
        return const_i64(ctx, &[v], &[]);
    }
    let v = ctx.emit(Prim::DimValues { exprs: vec![d] }, &[])?;
    reshape(ctx, v, vec![])
}

/// Row-major strides for `dims` (as dim expressions).
pub(crate) fn strides(dims: &[DimExpr]) -> Vec<DimExpr> {
    let mut s = vec![c(1); dims.len()];
    for i in (0..dims.len().saturating_sub(1)).rev() {
        s[i] = s[i + 1].clone() * dims[i + 1].clone();
    }
    s
}

/// Gather elements of `data` (any rank) by *linear* i64 indices `lin`
/// (any shape): flattens `data` and gathers on axis 0, so the output has
/// `lin`'s shape.
pub(crate) fn linear_gather(ctx: &mut LowerCtx, data: ValueId, lin: ValueId) -> Result<ValueId> {
    let n = prod(&dims(ctx, data));
    let flat = reshape(ctx, data, vec![n])?;
    ctx.emit(Prim::Gather { axis: 0 }, &[flat, lin])
}

/// Linear index tensor for "coordinates = iota on every axis except
/// `axis`, where the coordinate is `idx`" over `idx`'s shape, into a
/// tensor of shape `data_dims` (must be `idx`'s rank). This is the core of
/// GatherElements / ScatterElements / ReverseSequence.
pub(crate) fn linear_index_with_axis(
    ctx: &mut LowerCtx,
    data_dims: &[DimExpr],
    idx: ValueId,
    axis: usize,
) -> Result<ValueId> {
    let r = data_dims.len();
    let idx_dims = dims(ctx, idx);
    let st = strides(data_dims);
    let mut lin: Option<ValueId> = None;
    for d in 0..r {
        let coord = if d == axis {
            wrap_negative(ctx, idx, data_dims[d].clone())?
        } else {
            iota_along(ctx, idx_dims[d].clone(), r, d, DataType::I64)?
        };
        let term = if st[d] == c(1) {
            coord
        } else {
            let s = dim_value(ctx, st[d].clone())?;
            mul(ctx, coord, s)?
        };
        lin = Some(match lin {
            None => term,
            Some(acc) => add(ctx, acc, term)?,
        });
    }
    let lin = lin.expect("rank >= 1");
    // Ensure full shape (a rank-0 idx with r == 0 cannot happen).
    broadcast(ctx, lin, idx_dims)
}

/// Move `axis` to the last position; returns the value and the inverse
/// permutation to restore the layout.
pub(crate) fn axis_to_last(
    ctx: &mut LowerCtx,
    x: ValueId,
    axis: usize,
) -> Result<(ValueId, Vec<usize>)> {
    let r = rank(ctx, x);
    let mut perm: Vec<usize> = (0..r).filter(|&i| i != axis).collect();
    perm.push(axis);
    let mut inv = vec![0; r];
    for (i, &p) in perm.iter().enumerate() {
        inv[p] = i;
    }
    Ok((transpose(ctx, x, &perm)?, inv))
}

/// Record the single output.
pub(crate) fn out(ctx: &mut LowerCtx, v: ValueId) -> Result<()> {
    ctx.set_value(0, v);
    Ok(())
}

/// Numerically-safe softplus: `max(x, 0) + log1p(exp(-|x|))`.
pub(crate) fn softplus(ctx: &mut LowerCtx, x: ValueId) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let zero = scalar(ctx, dt, 0.0)?;
    let one = scalar(ctx, dt, 1.0)?;
    let ax = unary(ctx, UnaryOp::Abs, x)?;
    let nax = unary(ctx, UnaryOp::Neg, ax)?;
    let e = unary(ctx, UnaryOp::Exp, nax)?;
    let e1 = add(ctx, e, one)?;
    let l = unary(ctx, UnaryOp::Log, e1)?;
    let m = max(ctx, x, zero)?;
    add(ctx, m, l)
}

pub(crate) fn sigmoid(ctx: &mut LowerCtx, x: ValueId) -> Result<ValueId> {
    let dt = dtype(ctx, x);
    let one = scalar(ctx, dt, 1.0)?;
    let nx = unary(ctx, UnaryOp::Neg, x)?;
    let e = unary(ctx, UnaryOp::Exp, nx)?;
    let d = add(ctx, one, e)?;
    div(ctx, one, d)
}

/// Max-shifted log-softmax over `axes`.
pub(crate) fn log_softmax_axes(ctx: &mut LowerCtx, x: ValueId, axes: &[usize]) -> Result<ValueId> {
    let m = reduce(ctx, ReduceOp::Max, x, axes, true)?;
    let sh = sub(ctx, x, m)?;
    let e = unary(ctx, UnaryOp::Exp, sh)?;
    let s = reduce(ctx, ReduceOp::Sum, e, axes, true)?;
    let ls = unary(ctx, UnaryOp::Log, s)?;
    sub(ctx, sh, ls)
}

/// ArgMax/ArgMin along `axis` → i64, `keepdims` honored.
pub(crate) fn arg_reduce(
    ctx: &mut LowerCtx,
    x: ValueId,
    axis: usize,
    keepdims: bool,
    is_max: bool,
    select_last: bool,
) -> Result<ValueId> {
    let r = rank(ctx, x);
    let d = dims(ctx, x)[axis].clone();
    let op = if is_max { ReduceOp::Max } else { ReduceOp::Min };
    let m = reduce(ctx, op, x, &[axis], true)?;
    let eq = cmp(ctx, CmpOp::Eq, x, m)?;
    let idx = iota_along(ctx, d.clone(), r, axis, DataType::I64)?;
    let out = if select_last {
        let neg = scalar(ctx, DataType::I64, -1.0)?;
        let cand = select(ctx, eq, idx, neg)?;
        reduce(ctx, ReduceOp::Max, cand, &[axis], keepdims)?
    } else {
        // Sentinel = the axis length: above every valid index, and small
        // enough for backends that store i64 as 32-bit.
        let big = dim_value(ctx, d)?;
        let cand = select(ctx, eq, idx, big)?;
        reduce(ctx, ReduceOp::Min, cand, &[axis], keepdims)?
    };
    Ok(out)
}

/// Constant-mode padding of one axis: `before`/`after` may be negative
/// (crop). `value` is a rank-0 constant of the data dtype.
pub(crate) fn pad_axis_const(
    ctx: &mut LowerCtx,
    x: ValueId,
    axis: usize,
    before: i64,
    after: i64,
    value: ValueId,
) -> Result<ValueId> {
    let mut v = x;
    let d = dims(ctx, v);
    // Crops first.
    let mut start = c(0);
    let mut end = d[axis].clone();
    if before < 0 {
        start = c(before.unsigned_abs());
    }
    if after < 0 {
        end = end - c(after.unsigned_abs());
    }
    if before < 0 || after < 0 {
        v = slice_axis_dyn(ctx, v, axis, start, end)?;
    }
    let mut parts = Vec::new();
    if before > 0 {
        let mut bd = dims(ctx, v);
        bd[axis] = c(before as u64);
        parts.push(broadcast(ctx, value, bd)?);
    }
    parts.push(v);
    if after > 0 {
        let mut ad = dims(ctx, v);
        ad[axis] = c(after as u64);
        parts.push(broadcast(ctx, value, ad)?);
    }
    concat(ctx, &parts, axis)
}

/// The largest finite / infinite sentinel of a dtype, for pad values.
pub(crate) fn lowest(dt: DataType) -> f64 {
    match dt {
        DataType::F32 | DataType::F16 => f64::NEG_INFINITY,
        DataType::I64 => i64::MIN as f64,
        DataType::I32 => i32::MIN as f64,
        DataType::I8 => i8::MIN as f64,
        _ => 0.0,
    }
}
