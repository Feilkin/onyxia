//! Linear quantization ops over the 8-bit dtypes the IR has (`U8`/`I8`),
//! computed in f32 / i32 with explicit casts. Blocked and per-axis scales
//! expand to the input shape with reshape + broadcast.

use super::helpers::*;
use crate::{LowerCtx, LoweringRegistry, convert_proto_dtype};
use onyxia_ir::prim::{ReduceOp, UnaryOp};
use onyxia_ir::{DataType, DimExpr, Error, Result, ValueId};

pub(crate) fn register(r: &mut LoweringRegistry) {
    r.register("", "QuantizeLinear", quantize_linear);
    r.register("", "DequantizeLinear", dequantize_linear);
    r.register("", "DynamicQuantizeLinear", dynamic_quantize_linear);
    r.register("", "MatMulInteger", matmul_integer);
    r.register("", "QLinearMatMul", qlinear_matmul);
}

fn qrange(dt: DataType) -> Result<(f64, f64)> {
    Ok(match dt {
        DataType::U8 => (0.0, 255.0),
        DataType::I8 => (-128.0, 127.0),
        DataType::U4 => (0.0, 15.0),
        DataType::I4 => (-8.0, 7.0),
        other => return Err(Error::Unsupported(format!("quantized dtype {other}"))),
    })
}

/// Reshape a scale/zero-point to broadcast against `x` of `xd` dims:
/// scalar, per-axis `[Di]`, or blocked `[.., ceil(Di/B), ..]`.
fn expand_param(
    ctx: &mut LowerCtx,
    p: ValueId,
    xd: &[DimExpr],
    axis: usize,
    block: i64,
) -> Result<ValueId> {
    let pd = dims(ctx, p);
    let r = xd.len();
    if pd.is_empty() || (pd.len() == 1 && pd[0] == c(1) && r != 1) {
        return reshape(ctx, p, vec![]);
    }
    if pd.len() == 1 && r > 1 {
        let mut s = vec![c(1); r];
        s[axis] = pd[0].clone();
        return reshape(ctx, p, s);
    }
    if block > 0 && pd.len() == r {
        // [.., nb, ..] → [.., nb, 1, ..] → [.., nb, B, ..] → [.., nb*B, ..] → trim.
        let nb = pd[axis].clone();
        let mut s1 = pd.clone();
        s1.insert(axis + 1, c(1));
        let v = reshape(ctx, p, s1.clone())?;
        let mut s2 = s1;
        s2[axis + 1] = c(block as u64);
        let v = broadcast(ctx, v, s2)?;
        let mut s3 = pd.clone();
        s3[axis] = nb * c(block as u64);
        let v = reshape(ctx, v, s3)?;
        let d = xd[axis]
            .as_const()
            .ok_or_else(|| Error::Unsupported("blocked quantization on a symbolic axis".into()))?;
        return slice_axis(ctx, v, axis, 0, d);
    }
    Ok(p)
}

fn quantize_core(
    ctx: &mut LowerCtx,
    x: ValueId,
    scale: ValueId,
    zp: Option<ValueId>,
    qdt: DataType,
    axis: usize,
    block: i64,
) -> Result<ValueId> {
    let xd = dims(ctx, x);
    let (lo, hi) = qrange(qdt)?;
    let xf = cast(ctx, x, DataType::F32)?;
    let s = cast(ctx, scale, DataType::F32)?;
    let s = expand_param(ctx, s, &xd, axis, block)?;
    let q = div(ctx, xf, s)?;
    let q = unary(ctx, UnaryOp::Round, q)?;
    let q = match zp {
        Some(z) => {
            let z = cast(ctx, z, DataType::F32)?;
            let z = expand_param(ctx, z, &xd, axis, block)?;
            add(ctx, q, z)?
        }
        None => q,
    };
    let lo = scalar(ctx, DataType::F32, lo)?;
    let hi = scalar(ctx, DataType::F32, hi)?;
    let q = max(ctx, q, lo)?;
    let q = min(ctx, q, hi)?;
    cast(ctx, q, qdt)
}

fn quantize_linear(ctx: &mut LowerCtx) -> Result<()> {
    let (x, scale) = (val(ctx, 0)?, val(ctx, 1)?);
    let zp = opt_val(ctx, 2)?;
    let r = rank(ctx, x);
    let axis = ctx
        .norm_axis(ctx.attr_i("axis").unwrap_or(1), r.max(1))
        .unwrap_or(0);
    let block = ctx.attr_i("block_size").unwrap_or(0);
    let qdt = match (zp, ctx.attr_i("output_dtype")) {
        (Some(z), _) => dtype(ctx, z),
        (None, Some(d)) if d != 0 => convert_proto_dtype(d)?,
        _ => DataType::U8,
    };
    let y = quantize_core(ctx, x, scale, zp, qdt, axis, block)?;
    out(ctx, y)
}

fn dequantize_core(
    ctx: &mut LowerCtx,
    x: ValueId,
    scale: ValueId,
    zp: Option<ValueId>,
    axis: usize,
    block: i64,
    odt: DataType,
) -> Result<ValueId> {
    let xd = dims(ctx, x);
    let xf = cast(ctx, x, DataType::F32)?;
    let xf = match zp {
        Some(z) => {
            let z = cast(ctx, z, DataType::F32)?;
            let z = expand_param(ctx, z, &xd, axis, block)?;
            sub(ctx, xf, z)?
        }
        None => xf,
    };
    let s = cast(ctx, scale, DataType::F32)?;
    let s = expand_param(ctx, s, &xd, axis, block)?;
    let y = mul(ctx, xf, s)?;
    cast(ctx, y, odt)
}

fn dequantize_linear(ctx: &mut LowerCtx) -> Result<()> {
    let (x, scale) = (val(ctx, 0)?, val(ctx, 1)?);
    let zp = opt_val(ctx, 2)?;
    let r = rank(ctx, x);
    let axis = ctx
        .norm_axis(ctx.attr_i("axis").unwrap_or(1), r.max(1))
        .unwrap_or(0);
    let block = ctx.attr_i("block_size").unwrap_or(0);
    let odt = match ctx.attr_i("output_dtype") {
        Some(d) if d != 0 => convert_proto_dtype(d)?,
        _ => dtype(ctx, scale),
    };
    let y = dequantize_core(ctx, x, scale, zp, axis, block, odt)?;
    out(ctx, y)
}

fn dynamic_quantize_linear(ctx: &mut LowerCtx) -> Result<()> {
    let x = val(ctx, 0)?;
    let dt = dtype(ctx, x);
    let r = rank(ctx, x);
    let all: Vec<usize> = (0..r).collect();
    let zero = scalar(ctx, dt, 0.0)?;
    let mx = reduce(ctx, ReduceOp::Max, x, &all, false)?;
    let mx = max(ctx, mx, zero)?;
    let mn = reduce(ctx, ReduceOp::Min, x, &all, false)?;
    let mn = min(ctx, mn, zero)?;
    let range = sub(ctx, mx, mn)?;
    let q = scalar(ctx, dt, 255.0)?;
    let one = scalar(ctx, dt, 1.0)?;
    let is_zero = cmp(ctx, onyxia_ir::CmpOp::Eq, range, zero)?;
    let range = select(ctx, is_zero, one, range)?;
    let scale = div(ctx, range, q)?;
    let izp = div(ctx, mn, scale)?;
    let izp = unary(ctx, UnaryOp::Neg, izp)?;
    let izp = max(ctx, izp, zero)?;
    let izp = min(ctx, izp, q)?;
    let zpf = unary(ctx, UnaryOp::Round, izp)?;
    let y = div(ctx, x, scale)?;
    let y = unary(ctx, UnaryOp::Round, y)?;
    let y = add(ctx, y, zpf)?;
    let y = max(ctx, y, zero)?;
    let y = min(ctx, y, q)?;
    let y = cast(ctx, y, DataType::U8)?;
    let zp = cast(ctx, zpf, DataType::U8)?;
    ctx.set_value(0, y);
    ctx.set_value_opt(1, scale);
    ctx.set_value_opt(2, zp);
    Ok(())
}

/// Zero point broadcast for per-row (`a`) / per-column (`b`) vectors.
fn zp_for(ctx: &mut LowerCtx, zp: ValueId, x: ValueId, per_row: bool) -> Result<ValueId> {
    let z = cast(ctx, zp, DataType::I32)?;
    let zd = dims(ctx, z);
    let r = rank(ctx, x);
    if zd.len() == 1 && zd[0] != c(1) && r >= 2 {
        let mut s = vec![c(1); r];
        if per_row {
            s[r - 2] = zd[0].clone();
        } else {
            s[r - 1] = zd[0].clone();
        }
        return reshape(ctx, z, s);
    }
    if zd.len() == 1 {
        return reshape(ctx, z, vec![]);
    }
    Ok(z)
}

fn matmul_integer(ctx: &mut LowerCtx) -> Result<()> {
    let (a, b) = (val(ctx, 0)?, val(ctx, 1)?);
    let mut ai = cast(ctx, a, DataType::I32)?;
    let mut bi = cast(ctx, b, DataType::I32)?;
    if let Some(z) = opt_val(ctx, 2)? {
        let z = zp_for(ctx, z, a, true)?;
        ai = sub(ctx, ai, z)?;
    }
    if let Some(z) = opt_val(ctx, 3)? {
        let z = zp_for(ctx, z, b, false)?;
        bi = sub(ctx, bi, z)?;
    }
    let y = matmul(ctx, ai, bi)?;
    out(ctx, y)
}

/// QLinearMatMul: dequantize → matmul in f32 → quantize.
fn qlinear_matmul(ctx: &mut LowerCtx) -> Result<()> {
    let (a, a_scale, a_zp) = (val(ctx, 0)?, val(ctx, 1)?, val(ctx, 2)?);
    let (b, b_scale, b_zp) = (val(ctx, 3)?, val(ctx, 4)?, val(ctx, 5)?);
    let (y_scale, y_zp) = (val(ctx, 6)?, val(ctx, 7)?);
    let ra = rank(ctx, a);
    let rb = rank(ctx, b);
    let af = dequantize_core(
        ctx,
        a,
        a_scale,
        Some(a_zp),
        ra.saturating_sub(2),
        0,
        DataType::F32,
    )?;
    let bf = dequantize_core(
        ctx,
        b,
        b_scale,
        Some(b_zp),
        rb.saturating_sub(1),
        0,
        DataType::F32,
    )?;
    let yf = matmul(ctx, af, bf)?;
    let ry = rank(ctx, yf);
    let qdt = dtype(ctx, y_zp);
    let y = quantize_core(ctx, yf, y_scale, Some(y_zp), qdt, ry.saturating_sub(2), 0)?;
    out(ctx, y)
}
